import os
import csv
import json
import logging
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import datasets
from pprint import pprint
import azureml.core
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
import azureml.dataprep as dprep
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.webservice import AksWebservice
from utils.data_utils import *
from vision_evaluation.evaluators import MeanAveragePrecisionEvaluatorForSingleIOU


class Predictor:
    def __init__(self, 
                 workspace_name = None,
                 subscription_id = None,
                 resource_group = None,
                 aks_service_name = None,
                 scoring_uri = None,
                 key = None):
        if scoring_uri:
            self.setup_scoring(scoring_uri, key)
        elif workspace_name is not None and \
             subscription_id is not None and \
             resource_group is not None and \
             aks_service_name is not None:
            self.setup_azure(workspace_name,
                         subscription_id,
                         resource_group,
                         aks_service_name)
        else:
            logging.error("Must provide either scoring_uri"
                          "or all of (workspace_name, subscription_id,"
                          "resource_group, aks_service_name)")
        
    def setup_scoring(self,
                      scoring_uri,
                      key):
        headers = {'Content-Type': 'application/octet-stream'}
        if key is not None:
            headers['Authorization'] = f'Bearer {key}'
            
        self.request_func = partial(requests.post, 
                            url = scoring_uri,
                            headers = headers)
        
    def setup_azure(self,
                    workspace_name: str,
                    subscription_id: str,
                    resource_group: str,
                    aks_service_name: str):
        ws = Workspace.get(name=workspace_name,
                       subscription_id=subscription_id,
                       resource_group=resource_group)
        aks_service = AksWebservice(ws, aks_service_name)
        scoring_uri = aks_service.scoring_uri
        key, _ = aks_service.get_keys()

        headers = {'Content-Type': 'application/octet-stream',
                    'Authorization': f'Bearer {key}'}
        self.request_func = partial(requests.post, 
                            url = scoring_uri,
                            headers = headers)
        
    def convert_to_prediction_bbox(self, results,  width: int, height: int) -> List:
        pred = []
        for prediction in results['boxes']:
            xmin = prediction['box']['topX'] * width
            ymin = prediction['box']['topY'] * height
            xmax = prediction['box']['bottomX'] * width
            ymax = prediction['box']['bottomY'] * height
            probability = prediction['score']
            label = prediction['label']
            pred.append([width, height, xmin, ymin, xmax, ymax, label, probability])
        return pred

    def predict(self, local_fname):
        try:
            width, height = Image.open(local_fname).size
            with open(local_fname, mode="rb") as test_data:
                results = self.request_func(data = test_data)
                results = json.loads(results.text)
                results = self.convert_to_prediction_bbox(results,  width, height)
                return results
        except Exception as e:
            print("BUG", local_fname, e)

    def get_predictions(self,
                        image_dir: str,
                        fnames) -> dict:
        df_data = []
        failed_fnames = []
        for fname in tqdm(fnames):
            local_fname = os.path.join(image_dir, fname)
            results = self.predict(local_fname)
            if results is not None:
                for result in results:
                    df_data.append([fname] + result)
            else:
                failed_fnames.append(fname)
                        
        print("Failed files", failed_fnames)
        df = pd.DataFrame(df_data, 
                          columns = ["fname", 
                                     "width", "height", 
                                     "topX", "topY", 
                                     "bottomX", "bottomY", 
                                     "label", "Probability"])
    
        return df
    
class Evaluator:
    def __init__(self, jsonl: str,
                       pred_df: pd.DataFrame,
                       image_dir = "./",
                       predictor = None):
        if predictor is None and pred_df is None:
            logging.error("must provide either predictor or results_df!")
        
        self.setup_gts(jsonl)
        if pred_df is not None:
            self.pred_df = pred_df
        else:
            self.setup_predictions(predictor, image_dir)
        
        self.setup_concat_df()
        self.image_dir = "./"
        self.visualizer = InputVisualizerOD(self.concat_df["label"].unique())

    def setup_gts(self, jsonl):
        parser =  JsonParser("OD")
        self.gt_df = parser.parse(jsonl)
        
    def setup_predictions(self, jsonl, image_dir):
        fnames = gt_df["fname"].unique()
        pred_df = self.predictor.get_predictions(image_dir, fnames)
        self.pred_df = pred_df
        
    def setup_concat_df(self):
        # concatenate the two
        self.concat_df = pd.concat([self.gt_df, self.pred_df]).reset_index(drop = True)
        labels = self.concat_df["label"].unique()
        label_mapping = {l:i for l, i in zip(labels, range(len(labels)))}
        self.concat_df["label_encode"] = self.concat_df["label"].apply(lambda x: label_mapping[x])
        
    def get_mAP(self, iou = 0.5, threshold = 0.5):
    
        mAP_rows = []
        for fname in self.pred_df["fname"].unique():
            subdf = self.concat_df[self.concat_df["fname"] == fname]
            groundtruths = subdf[subdf["Probability"].isna() & (subdf["isCrowd"] == False)][
                ["label_encode", "topX", "topY", "bottomX", "bottomY"]].\
                values.tolist()
            
            predictions = subdf[~subdf["Probability"].isna() & 
                                  subdf["Probability"] > threshold][
                                    ["label_encode", "Probability", 
                                     "topX", "topY", "bottomX", "bottomY"]].values.tolist()
            evaluator = MeanAveragePrecisionEvaluatorForSingleIOU(
                        iou = iou)
            evaluator.add_predictions([predictions], 
                                      [groundtruths])
            report = evaluator.get_report()
            
            report["fname"] = fname
            mAP_rows.append(report)
            
        self.image_df = pd.DataFrame(mAP_rows).\
                            sort_values(f"mAP_{int(iou * 100)}")
        self.threshold = threshold

    def visualize_by_fname(self, fname):
        image_df = self.concat_df[self.concat_df["fname"] == fname]
        image_df = image_df[~(image_df["Probability"] < self.threshold)]
        self.visualizer.visualize(image_fname = fname, image_df = image_df)

    def random_visualize(self, k = 1, from_worst_mAP = False):
        if from_worst_mAP:
            subdf = self.image_df.head(k)
        else:
            subdf = self.image_df.sample(k, random_state = 42)
        
        for i, (_, row) in enumerate(subdf.iterrows()):
            print(i, row)
            self.visualize_by_fname(row["fname"])