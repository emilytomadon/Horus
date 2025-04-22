# FALCON: Fair Face Recognition via Local Optimal Feature Normalization
Paper of code: [https://openaccess.thecvf.com/content/WACV2025/html/Al-Refai_FALCON_Fair_Face_Recognition_via_Local_Optimal_Feature_Normalization_WACV_2025_paper.html](https://openaccess.thecvf.com/content/WACV2025/html/Al-Refai_FALCON_Fair_Face_Recognition_via_Local_Optimal_Feature_Normalization_WACV_2025_paper.html)

## Information to the code base of FALCON and the corresponding experiments


### Overview over the (important) files
- Folder *Data* should hold all files regarding the database information. For each investigated face recognition system and database is holds the pre-processed embeddings, the filenames, the identity information and the attribute labels for each sample.
    - It should have the following structure:
        - *Data*:
            - *[FRSystem]*:
                - *[Dataset]*:
                    - emb.npy: embeddings of the face recognition system
                    - filenames.npy: filenames of the images
                    - identities.npy: identity information of the images
                    - labels_{age/gender/...}.npy: attribute labels of the images
- Folders *Baseline*, *FairCal*, *FALCON*, *FSN*, *FTC* and *SLF* each hold a Python file for executing the corresponding method. After executing the experiments, it will also contain a folder *result* with all the `Result` objects that contain the evaluation results after applying the method on several settings
- Folder *helper_classes* holds several files that provide classes which are helpful for applying the methods
    - File *dataset.py* provides a `Database` class which supplies all information and files regarding a specific database
    - File *fairness_approach.py* provides the base class `FairnessApproach` all methods *Baseline*, *FairCal*, *FALCON*, *FSN*, *FTC* and *SLF* inherit from. It contains the interface methods and the common methods.
    - File *result.py* provides a class `Result` which is an object that contains and makes available all evaluation results in a structured way.
    - File *dataset_infos.py* does not provide a class but information regarding a specific `Database` object regarding the distribution of the labels for each attribute
- Folder *tools* holds several files that are helpful for the methods and their evaluation
    - File *enums.py* holds all enumerations that are used within this thesis. It holds enumerations for the `FRSystem`, the `Dataset`, the `Datatype` (embeddings, filenames or identities of datasets), the `Attribute`, the `Metric` and the `Method`.
    - File *fairness_evaluation.py* holds the methods for calculating the fairness metrics `FDR`, `IR` and `GARBE`.
    - File *fnmr_fmr.py* holds the methods to calculate the False Non Match Rates, False Match Rates and thresholds depending on labels, scores and either a fixed FMR or a fixed threshold.
    - File *group_scores.py* provides methods to access the files in folder *Data*.
- Folder experiments holds the file for the parameter experiment
    - File *baseline_visualization.py* provides methods to visualise the optimal local thresholds, before and after applying FALCON in a plot
    - File *parameter_experiment.py* executes the parameter experiment analyzing the trade-off between fairness and performance in face recognition systems using various parameter combinations.
    - File *visualize-FSN-FALCON.py* provides methods to create images for visualize the difference between FSN and FALCON as well as the clustering of FSN using a Voronoi diagram.
- Folder *timing* holds the results of the timing experiment
- File *main.py* is the main file that is used to execute the experiments.

## Citing
If you use this code, please cite the following paper:
```
@InProceedings{Al-Refai_2025_WACV,
    author    = {Al-Refai, Rouqaiah and Hempel, Philipp and Biagi, Clara and Terh\"orst, Philipp},
    title     = {FALCON: Fair Face Recognition via Local Optimal Feature Normalization},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {3416-3426}
}
```

## License
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

Copyright (c) 2025 Philipp Hempel