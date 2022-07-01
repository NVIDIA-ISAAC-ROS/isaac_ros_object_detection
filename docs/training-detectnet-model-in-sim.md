## Training a model using simulation

There are multiple ways to train your own `Detectnet_v2` base model. Note that you will need to update parameters, launch files, and more to match your specific trained model.

### Use the TAO toolkit launcher
The `Train and Optimize` tookit from NVIDIA has all the tools you need to prepare a dataset and re-train a detector with an easy to follow Jupyter notebook tutorial.

1. Install the `tao` command line utilities
   ```bash
   pip3 install jupyterlab nvidia-pyindex nvidia-tao
   ```
2. Obtain an [NGC API key](https://ngc.nvidia.com/setup/api-key).
3. Install and configure `ngc cli` from [NVIDIA NGC CLI Setup](https://ngc.nvidia.com/setup/installers/cli).
   ```bash
     wget -O ngccli_linux.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip && unzip -o ngccli_linux.zip && chmod u+x ngc && \
     md5sum -c ngc.md5 && \
     echo "export PATH=\"\$PATH:$(pwd)\"" >> ~/.bash_profile && source ~/.bash_profile && \
     ngc config set
   ```
4. Download the TAO cv examples to a local folder
   ```bash
     ngc registry resource download-version "nvidia/tao/cv_samples:v1.3.0"
   ```
5. Run the `DetectNet_v2` Jupyter notebook server.
   ```bash
     cd cv_samples_vv1.3.0 && jupyter-notebook --ip 0.0.0.0 --port 8888 --allow-root
   ```
6. Navigate to the DetectNet v2 notebook in `detectnet_v2/detectnet_v2.ipynb` or go to
   ```
     http://0.0.0.0:8888/notebooks/detectnet_v2/detectnet_v2.ipynb
   ```
   And follow the instructions on the tutorial.

### Training object detection in simulation

If you wish to generate training data from simulation using 3D models of the object classes you would like to detect, consider following the tutorial [Training Object detection from Simulation](https://docs.nvidia.com/isaac/doc/tutorials/training_in_docker.html).

The tutorial will use simulation to create a dataset that can then be used to train a `DetectNet_v2` based model. It's an easy to use tool with full access to customize training parameters in a Jupyter notebook.

Once you follow through the tutorial, you should have an `ETLT` file in `~/isaac-experiments/tlt-experiments/experiment_dir_final/resnet18_detector.etlt`.

Consult the spec file in `~/isaac-experiments/specs/isaac_detect_net_inference.json` for the values to use in the following section when preparing the model for usage with this package.

### Using the included dummy model for testing

In this package, you will find a pre-trained DetectNet model that was trained solely for detecting tennis balls using the described simulation method. Please use this model only for verification or exploring the pipeline.

> **Note**: Do not use this tennis ball detection model in a production environment.

You can find the `ETLT` file in `isaac_ros_detectnet/test/dummy_model/detectnet/1/resnet18_detector.etlt` and use the ETLT key `"object-detection-from-sim-pipeline"`, including the double quotes.

```bash
export PRETRAINED_MODEL_ETLT_KEY=\"object-detection-from-sim-pipeline\"
```
