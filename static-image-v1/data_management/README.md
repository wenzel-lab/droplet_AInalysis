# Data Management Module

This folder contains helper modules and classes for managing, processing, and analyzing droplet data in the static-image-v1 workflow. These scripts are essential for handling the extraction, aggregation, and statistical analysis of droplet features detected in microscopy images.

## Contents

- **ImageData.py**  
  Defines the `ImageData` class for storing and aggregating droplet statistics, distributions, and batch analysis results.

- **data_tools.py**  
  Utility functions for mathematical operations, grouping, and formatting data for further analysis.

- **get_boxes.py**  
  Extracts bounding box coordinates from YOLO model results, draws annotations on images, and saves the annotated outputs.

- **get_distributions.py**  
  Computes droplet size and volume distributions from YOLO detection results, using the above tools and classes.

## Purpose

These modules are used by the main analysis pipeline to:
- Aggregate and store droplet measurements
- Annotate images with detection results
- Compute and group statistical distributions

**Do not delete this folder if you want to maintain or extend the data processing functionality of the project.**

---
