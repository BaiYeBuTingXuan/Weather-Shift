# Params
********************************
Time: 2024-03-21 13:06:26
**Description**: train classifier of 8892
| Param | Value | Description |
| ----- | ----- | ----------- |
|**train_name**|8892|list of train lidar|
|**lidar**|['lidar_hdl64_strongest']|list of train lidar|
|**dataset_path**|/home/wanghejun/Desktop/wanghejun/WeatherShift/main/data/Dense/SeeingThroughFog|path of the dataset|
|**batch_size**|32|size of the batches|
|**epoch**|0|epoch to start training from|
|**n_epochs**|1000|number of epochs of training|
|**n_cpu**|16|number of CPU threads to use during batches generating|
|**lr**|5e-08|learning rate|
|**weight_decay**|0.0005|adam: weight_decay|
|**checkpoints_interval**|100|interval between model checkpoints|
|**clip_value**|1|Clip value for training to avoid gradient explosion|
********************************

