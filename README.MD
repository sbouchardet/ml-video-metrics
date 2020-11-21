# ML Video Metrics

ML Video Metricis is a tool to collect metrics of ML experiments in videos.

# Install

This project uses [pipenv](https://pypi.org/project/pipenv/) as a dependencies maneger.

Assuming you already have pipenv intalled, you should intall the project with:

`pipenv install`

This command will create an virtual env ans intall all dependencies.

# How to use

If you install the project correctly, you could start the shell section in the virtual env with:

`pipenv shell`

After this, you could run the command:

`ml-video-metrics --help`

This command output is

```shell
Usage: ml-video-metrics [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  similarity
  video-object-segmentation
```

It shows the kinds of metrics. 

## Video Object Segmentation metrics
The command `ml-video-metrics video-object-segmentation --help` will return a help test to the user. The output of this command is

```shell
Options:
  -k, --kinds [precision|recall|IoU]
                                  Metric kind
  -t, --true TEXT                 Path to ground truth binary masks
  -p, --predicted TEXT            Path to the predicted binary masks
  -v, --video-name TEXT           Video name (file looked up inside the paths
                                  --true and --predicted)

  -o, --output TEXT               Output file
  --help                          Show this message and exit.
```
This command parameters required are:
 - `kinds`: This current implementation suport only the `precision`, `recall` and `IoU` value.

 - `true`: This param should be the path to the video folders with the true value os frames. The frames used as reference.

 - `predicted`: This param should be the path to the video folders with the predicted value os frames. The result of the experiment.

 - `video-name`: This is a reference to the folder name inside the `true` and `predicted` path.

This command optional parametes are:
- `output`: This parameter expects a output file name. The default value is `out.json`.

## Similarity metrics

The command `ml-video-metrics similarity --help` will return a help test to the user. The output of this command is

```shell
Usage: ml-video-metrics similarity [OPTIONS]

Options:
  -k, --kinds [structural-similarity]
                                  Metric kind
  -t, --true TEXT                 Path to ground truth frames
  -p, --predicted TEXT            Path to the predicted frames
  -v, --video-name TEXT           Video name (file looked up inside the paths
                                  --true and --predicted)

  -o, --output TEXT               Output folder
  --save-extra / --no-save-extra
  --help                          Show this message and exit.
```

This command parameters required are:
 - `kinds`: This current implementation suport only the `structural-similarity` value.

 - `true`: This param should be the path to the video folders with the true value os frames. The frames used as reference.

 - `predicted`: This param should be the path to the video folders with the predicted value os frames. The result of the experiment.

 - `video-name`: This is a reference to the folder name inside the `true` and `predicted` path.

This command optional parametes are:
- `output`: This parameter expects a output folder. In this metric, since we can have a map of similarity of the image (that is an image file per frame), the results will be persested in a folder. The default value is `./out`.
- `save-extra`: If this flac is activate, this command will save the similarity maps in the output folder. Otherwise, the output will be only a json inside the folder with the metrics values.