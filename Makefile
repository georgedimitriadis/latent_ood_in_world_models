.DEFAULT_GOAL := help
.PHONY: create_data train_models printout_models_layers visualise_saved_data visualise_result_curves visualise_all_models_for_some_samples visualise_copying clean lint create_environment help


PROFILE = default
PROJECT_NAME = latent_ood_in_world_models

# Each batch has 100 samples
NUM_TRAIN_BATCHES = 1000
NUM_TEST_BATCHES = 10
NUM_TRAINING_EPOCHS = 100
NUM_TRAIN_SET_IMAGES_TO_VISUALISE = 200
NUM_TEST_SET_IMAGES_TO_VISUALISE = 20

ifeq ($(OS),Windows_NT)
    detected_OS := Windows

    export PYTHON_INTERPRETER = python
	export CUDA_VISIBLE_DEVICE = "0"
	export BACKEND = jax

    export SET_CMD = set
    export AND_CMD = &
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')

    PYTHON_INTERPRETER = python
	CUDA_VISIBLE_DEVICE = "0"
	BACKEND =jax

    SET_CMD =
    AND_CMD =

    PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
	BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

    ifeq (,$(shell which conda))
		HAS_CONDA=False
	else
		HAS_CONDA=True
	endif
endif



#################################################################################
# COMMANDS                                                                      #
#################################################################################


# Targets
all: $(SO_FILE)


## Create all datasets
create_data:
	@echo "Making translate train set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TRAIN_BATCHES) data/processed/compositional_translate/train.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":1, "transformation_type": "translate"}'
	@echo "Making translate test distance 0 set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TEST_BATCHES) data/processed/compositional_translate/test_d0.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":1, "transformation_type": "translate"}'
	@echo "Making translate test distance 1 set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TEST_BATCHES) data/processed/compositional_translate/test_d1.npz generate_compositional_datasets '{"distance":1, "symmetric_objects":1, "transformation_type": "translate"}'
	@echo "Making translate test distance 2 set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TEST_BATCHES) data/processed/compositional_translate/test_d2.npz generate_compositional_datasets '{"distance":2, "symmetric_objects":1, "transformation_type": "translate"}'
	@echo "Making rotate train set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TRAIN_BATCHES) data/processed/compositional_rotate/train.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":0, "transformation_type": "rotate"}'
	@echo "Making rotate test distance 0 set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TEST_BATCHES) data/processed/compositional_rotate/test_d0.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":0, "transformation_type": "rotate"}'
	@echo "Making rotate test distance 1 set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TEST_BATCHES) data/processed/compositional_rotate/test_d1.npz generate_compositional_datasets '{"distance":1, "symmetric_objects":0, "transformation_type": "rotate"}'
	@echo "Making rotate test distance 2 set"
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py $(NUM_TEST_BATCHES) data/processed/compositional_rotate/test_d2.npz generate_compositional_datasets '{"distance":2, "symmetric_objects":0, "transformation_type": "rotate"}'


## Train all models
train_models:
	@echo "Train MLP on translate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures mlp_nn $(NUM_TRAINING_EPOCHS) saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train CNN on translate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures cnn $(NUM_TRAINING_EPOCHS) saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train Transformer on translate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures transformer $(NUM_TRAINING_EPOCHS) saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train Axial Pointer Full on translate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_pointer_network_full $(NUM_TRAINING_EPOCHS) saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train Axial Pointer Linear on translate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_pointer_network_lines $(NUM_TRAINING_EPOCHS) saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train MLP on rotate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures mlp_nn $(NUM_TRAINING_EPOCHS) saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train CNN on rotate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures cnn $(NUM_TRAINING_EPOCHS) saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train Transformer on rotate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures transformer $(NUM_TRAINING_EPOCHS) saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train Axial Pointer Full on rotate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_pointer_network_full $(NUM_TRAINING_EPOCHS) saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train Axial Pointer Linear on rotate"
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_pointer_network_lines $(NUM_TRAINING_EPOCHS) saved_models/rotate data/processed/compositional_rotate data/results/rotate


## Create Printouts of a models layers and their shapes
printout_models_layers:
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/printout_model.py saved_models/translate/axial_pointer_network_full.keras

## Create Images of the saved data sets
visualise_saved_data:
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_translate/train.npz data/results/translate/figures/train $(NUM_TRAIN_SET_IMAGES_TO_VISUALISE)
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_rotate/train.npz data/results/rotate/figures/train $(NUM_TRAIN_SET_IMAGES_TO_VISUALISE)
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_translate/test_d0.npz data/results/translate/figures/test_d0 $(NUM_TEST_SET_IMAGES_TO_VISUALISE)
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_rotate/test_d0.npz data/results/rotate/figures/test_d0 $(NUM_TEST_SET_IMAGES_TO_VISUALISE)
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_translate/test_d1.npz data/results/translate/figures/test_d1 $(NUM_TEST_SET_IMAGES_TO_VISUALISE)
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_rotate/test_d1.npz data/results/rotate/figures/test_d1 $(NUM_TEST_SET_IMAGES_TO_VISUALISE)
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_translate/test_d2.npz data/results/translate/figures/test_d2 $(NUM_TEST_SET_IMAGES_TO_VISUALISE)
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/visualization/visualise_training_data_sets.py data/processed/compositional_rotate/test_d2.npz data/results/rotate/figures/test_d2 $(NUM_TEST_SET_IMAGES_TO_VISUALISE)

## Create the Figure 3 image of the error curves
visualise_result_curves:
	#$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/analysis/scripts_for_images/figure_3_results.py data/results  data/results
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/analysis/scripts_for_images/figure_3_results.py "E:\Projects Large\Learning\TempData\ood_paper\logs_1"  data/results

## Create images, each one showing the result of all networks for a specific sample. Used to put together figure 4. Change the data type, the distance and the list of samples accordingly
visualise_all_models_for_some_samples:
	$(SET_CMD) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) $(AND_CMD) $(SET_CMD) KERAS_BACKEND=$(BACKEND) $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/analysis/scripts_for_images/figure_4_errors.py translate 0 "[10, 100, 200]" saved_models data/processed data/results/translate/all_models_samples


## Create images showing the copying of pixels from the first N samples of both networks on both data sets over all distances
visualise_copying:
	$(SET_CMD) CUDA_VISIBLE_DEVICES="" $(AND_CMD) $(SET_CMD) PYTHONPATH=./src $(AND_CMD) $(PYTHON_INTERPRETER) src/experiments/analysis/scripts_for_images/figure_5_copy_visualisations.py 50 saved_models data/processed data/results

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

