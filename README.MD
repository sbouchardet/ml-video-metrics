# ML Video Metrics

ML Video Metricis is a tool to collect metrics of ML experiments in videos.

# Install

This project uses [pipenv](https://pypi.org/project/pipenv/) as a dependencies maneger.

Assuming you already have pipenv intalled, you should intall the project with:

`pipenv install`

This command will create a virtual env and install all dependencies.

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

It shows the kinds of metrics that ML Video Metrics can calculate.

## Video Object Segmentation metrics
The command `ml-video-metrics video-object-segmentation --help` will return a help text to the user. The output of this command is

```shell
Usage: ml-video-metrics video-object-segmentation [OPTIONS]

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
The required command parameters are:
 - `kinds`: This current implementation supports only the `precision`, `recall` and `IoU` values.

 - `true`: This param should be the path to the video folder with the true value of the masks. The masks that will be used as reference.

 - `predicted`: This param should be the path to the video folder with the predicted value of the masks. The result of the experiment.

 - `video-name`: This is a reference to the folder name inside the `true` and the `predicted` path.

The optional command parameters are:
- `output`: This parameter expects an output file name. The default value is `out.json`.

## Similarity metrics

The command `ml-video-metrics similarity --help` will return a help text to the user. The output of this command is

```shell
Usage: ml-video-metrics similarity [OPTIONS]

Options:
  -k, --kinds [structural-similarity]
                                  Metric kind
  -t, --true TEXT                 Path to ground truth frames
  -p, --predicted TEXT            Path to the predicted frames
  -v, --video-name TEXT           Video name (the folder looked up inside the
                                  paths --true and --predicted)

  -o, --output TEXT               Output folder
  --save-extra / --no-save-extra
  --help                          Show this message and exit.
```

The required command parameters are:
 - `kinds`: This current implementation supports only the `structural-similarity` value.

 - `true`: This param should be the path to the video folders with the true value of frames. The frames used as reference.

 - `predicted`: This param should be the path to the video folders with the predicted value of frames. The result of the experiment.

 - `video-name`: This is a reference to the folder name inside the `true` and `predicted` path.

The optional command parametes are:
- `output`: This parameter expects an output folder. In this metric, since we can have the similarity-maps of the frames (that is an image file per frame), the results will be saved in a folder. The default value is `./out`.

- `save-extra` or `no-save-extra`: If this flag is active, this command will save the similarity-maps in the output folder. Otherwise, the output will be only a JSON file inside the folder with the metrics values. The default value is `save-extra`.
