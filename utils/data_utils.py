import os
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copy2
from typing import List, Dict
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, clear_output

def load_jsonl(json_fname: str) -> List[Dict]:
    lines = open(json_fname, encoding = "utf8").readlines()
    annotation_list = [json.loads(line) for line in lines]
    return annotation_list

def save_jsonl(json_fname: str, annotation_list: List):
    converted_lines = []
    for annotation_dict in annotation_list:
        converted_line = json.dumps(annotation_dict, ensure_ascii=False)
        converted_lines.append(converted_line)
    with open(json_fname, "w", encoding = "utf8") as f:
        f.write("\n".join(converted_lines))

def get_percentage_df(series, total):
    return pd.concat(
                    [series,
                    (series/total).\
                     apply(lambda x: format(x,'.2%')).\
                     rename("Percentage")],
                axis = 1)

class JsonParser:
    def __init__(self, task_type = ["OD", "MC", "ML", "SEG"]):
        """ parse json to dataframe for local analysis and visualization
        params: task_type: 
                OD: Object Detection;
                MC: Multi-class Classification
                ML: Multi-label Classification
                SEG: Instance Segmentation
        """
        self.parser = None
        if task_type == "OD":
            self.parser = self.convert_od_bboxes
        elif task_type == "MC":
            self.parser = self.convert_cls_label
        else:
            raise NotImplementedError("task type not supported!")

        
    def convert_od_bboxes(self, label: List, 
                                width: int,
                                height: int,
                                fname: str) -> List[List]:
        return {"fname": fname,
                "width": width,
                "height": height,
                "topX": label["topX"] * width, 
                "topY": label["topY"] * height, 
                "bottomX": label["bottomX"] * width, 
                "bottomY": label["bottomY"] * height,
                "label": label["label"],
                "isCrowd": label["isCrowd"] in {"True", "true", "1", 1}}

    def convert_cls_label(self,  label: List, 
                                width: int,
                                height: int,
                                fname: str) -> List[List]:
        return {"fname": fname,
                "width": width,
                "height": height,
                "label": label["label"]}
    
    def convert_single_image(self, annotation_dict: dict) -> dict:
        labels = []
        fname = annotation_dict["image_url"][len("AmlDatastore://"):]
        width = int(annotation_dict["image_details"]["width"][:-2])
        height = int(annotation_dict["image_details"]["height"][:-2])
        for label in annotation_dict["label"]:
            label = self.parser(label, width, height, fname)
            labels.append(label)
        return labels

    def parse(self, json_fname: str) -> dict:
        annotation_list = load_jsonl(json_fname)
        labels = []
        for annotation_dict in annotation_list:
            labels.extend(self.convert_single_image(annotation_dict))
        df = pd.DataFrame(data = labels)
        return df

class InputVisualizerOD:

    def __init__(self, labels = None):
        self.font = ImageFont.truetype("./arial.ttf", size = 20)
        
        self.label_to_color = dict()
        if labels is not None:
            self.get_distinct_colors(labels)
            
    def get_color(self, label):
        if label in self.label_to_color:
            return self.label_to_color[label]
        else:
            color = np.random.choice(range(256), size=3)
            self.label_to_color[label] = color
            return color
    
    def get_distinct_colors(self, labels):
        # a list of RGB tuples
        colors = sns.color_palette('husl', 
                                   n_colors = len(labels))  

        for label, color in zip(labels, colors):
            self.label_to_color[label] = tuple([int(c* 255) for c in color])
    
    def visualize(self, image_fname, image_df):
        image = Image.open(image_fname)
        draw = ImageDraw.Draw(image)

        for _, bbox in image_df.iterrows():
            color = self.get_color(bbox["label"])
            draw.rectangle((bbox["topX"], 
                            bbox["topY"], 
                            bbox["bottomX"], 
                            bbox["bottomY"]), 
                            outline = color, 
                            width = 5)
            draw.text((bbox["topX"], bbox["topY"] - 20),
                       bbox["label"], 
                       align ="left",
                       font = self.font,
                       fill = color)
        display(image)

class InputAnalyzer:
    def __init__(self, task_type = ["OD", "MC", "ML", "SEG"]):
        self.json_parser = JsonParser(task_type)
        self.dataframes = []
        self.dataframe = None
    
    ######### File Loading ##########
    def add_file(self, json_fname: str, train_type = ["train", "validation", "test"]):
        df = self.json_parser.parse(json_fname)
        df["train_type"] = train_type
        print(f"Added {train_type} dataframe with shape", df.shape)
        self.dataframes.append(df)
    
    def aggregate_image_dfs(self):
        print(f"Aggregating {len(self.dataframes)} dataframes.")
        self.image_df = pd.concat(self.dataframes, axis = 0)[
                            ["fname", "width", "height", "train_type", "label"]]
        self.image_df = self.image_df.groupby("fname").agg({
                                "width": "median",
                                "height": "median",
                                "train_type": "first",
                                "label": lambda x: list(set(x))
                            }).reset_index()
        display(self.image_df.head())
        
    def image_count_analysis(self):
        print("Total images: ", len(self.image_df))
        print("Analyzing images per train type:")
        display(get_percentage_df(
                series = self.image_df["train_type"].value_counts(),
                total = len(self.image_df)
                ))

    ######### Resolution Analysis ##########
    def image_dimension_plot(self):
        print("Creating image dimension plot")
        plt.figure(figsize=(10, 6))
        ax = sns.kdeplot(data = self.image_df, x = "width", y = "height",
                          cmap = "Reds", shade=True, bw_adjust=.5).\
                         set(title = "Height& Width 2D Plot")
        plt.xticks(rotation=45)
        plt.show()
        
    def image_dimension_analysis(self):    
        print("Analyzing images dimension across train_type:")
        display(self.image_df.aggregate(
            {"height": ["min", "max", "median"], 
             "width":  ["min", "max", "median"]}))

        print("Analyzing images dimension with train_type breakdown:")
        display(self.image_df.groupby("train_type").agg(
            {"height": ["min", "max", "median"], 
             "width":  ["min", "max", "median"]}))

    ######## label Analysis ##############
    def label_analysis(self):
        image_df_with_full_labels = \
                self.image_df.\
                     explode('label').\
                     fillna('')
        unique_labels = image_df_with_full_labels["label"].unique()
        print(f"{len(unique_labels)} Labels: ", ", ".join(unique_labels))
        
        print("Analyzing image count per label")
        display(get_percentage_df(
                    series = image_df_with_full_labels["label"].value_counts(),
                    total = len(self.image_df)
        ))

        
class InputAnalyzerOD (InputAnalyzer):
    def __init__(self):
        super().__init__(task_type = "OD")
        self.visualizer = InputVisualizerOD()
        random.seed(100)
        
    def aggregate_dfs(self):
        super().aggregate_image_dfs()
        
        print(f"Aggregating bbox level {len(self.dataframes)} dataframes.")
        self.bbox_df = pd.concat(self.dataframes, axis = 0)
        print("bbox_df",)
        display(self.bbox_df.head())
        
        # setups for visualizer
        self.visualizer.get_distinct_colors(self.bbox_df["label"].unique())
        
        # setups for size column
        def get_size(row):
            height, width = (row["bottomY"] - row["topY"]), (row["bottomX"] - row["topX"])
            size = height*width 
            if size < 32 * 32:
                size = "small (<32x32)"
            elif size > 96 * 96:
                size = "large (>96x96)"
            else:
                size = "medium"
            return height, width, size
        self.bbox_df[["bbox_height", "bbox_width", "bbox_size"]] = \
                            self.bbox_df.apply(get_size, axis = 1, result_type='expand')    
        
    def bbox_count_analysis(self):
        print("Total bounding boxes: ", len(self.bbox_df))
        print("Analyzing bboxes per train type:")
        display(get_percentage_df(
                    series = self.bbox_df["train_type"].value_counts(),
                    total = len(self.bbox_df)
                    ))
        
    def bbox_per_image_analysis(self):
        print(f"Analyzing bounding boxes per images")
        display(self.bbox_df["fname"].value_counts().\
                        aggregate(["min", "max", "median"]))
        
        print(f"Analyzing bounding boxes per images per train type:")
        bbox_per_images_df = self.bbox_df[["fname", "train_type"]].\
                            groupby(["train_type", "fname"]).\
                            agg({"fname": "count"}).\
                            rename(columns = {"fname": "bbox_per_image"}).\
                            reset_index()
        display(bbox_per_images_df.groupby("train_type").\
                        agg(["min", "max", "median"]))
    
    def bbox_size_analysis(self):
        print("Analyzing size per train type")
        display(self.bbox_df.pivot_table(
                     index = "train_type", 
                     columns = "bbox_size", 
                     values = "fname", 
                     aggfunc = "count").\
                     fillna(0))
       
        print("Analyzing size per label")
        display(self.bbox_df.pivot_table(
                                         index = "label", 
                                         columns = "bbox_size", 
                                         values = "fname", 
                                         aggfunc = "count").\
                             fillna(0))
    def bbox_dimension_plot(): 
        print("Creating bounding box dimension plot")
        g = sns.FacetGrid(self.bbox_df, 
                          col="label", col_wrap = 3,
                          sharex = False, sharey = False
                         )

        g.map(sns.kdeplot, "bbox_width", "bbox_height", 
                           shade = True, bw_adjust=.5)
        
        
    def bbox_label_analysis(self):
        print("Analyzing bbox count per label")

        display(get_percentage_df(
                    series = self.bbox_df["label"].value_counts(),
                    total = len(self.bbox_df)
                )) 
    
    def visualize_by_fname(self, fname):
        print(fname)
        self.visualizer.visualize(image_fname = fname,
                                  image_df = self.bbox_df[self.bbox_df["fname"] == fname])
    
    def random_visualize(self, k = 1, condition = dict()):
        sub_df = self.bbox_df
        for key, val in condition.items():
            sub_df = sub_df[sub_df[key] == val]
        fnames = random.choices(sub_df["fname"].unique(), k = k)
        for i, fname in enumerate(fnames):
            print(i,)
            self.visualize_by_fname(fname)