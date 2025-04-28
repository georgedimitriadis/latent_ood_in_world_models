.PHONY: createdata

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = latent_ood_in_world_models
PYTHON_INTERPRETER = python
CUDA_VISIBLE_DEVICE = "0"
BACKEND = "jax"

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


# Targets
all: $(SO_FILE)


## Create all datasets
create_data:
	@echo "Making translate train set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 1000 data/processed/compositional_translate/train.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":0, "transformation_type": "translate"}'
	@echo "Making translate test distance 0 set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_translate/test_d0.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":1, "transformation_type": "translate"}'
	@echo "Making translate test distance 1 set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_translate/test_d1.npz generate_compositional_datasets '{"distance":1, "symmetric_objects":1, "transformation_type": "translate"}'
	@echo "Making translate test distance 2 set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_translate/test_d2.npz generate_compositional_datasets '{"distance":2, "symmetric_objects":1, "transformation_type": "translate"}'
	@echo "Making rotate train set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 1000 data/processed/compositional_rotate/train.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":0, "transformation_type": "rotate"}'
	@echo "Making rotate test distance 0 set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_rotate/test_d0.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":0, "transformation_type": "rotate"}'
	@echo "Making rotate test distance 1 set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_rotate/test_d1.npz generate_compositional_datasets '{"distance":1, "symmetric_objects":0, "transformation_type": "rotate"}'
	@echo "Making rotate test distance 2 set"
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_rotate/test_d2.npz generate_compositional_datasets '{"distance":2, "symmetric_objects":0, "transformation_type": "rotate"}'


## Train all models
train_models:
	@echo "Train MLP on translate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures mlp_nn 100 saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train CNN on translate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures cnn 100 saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train Transformer on translate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures transformer 100 saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train Axial Pointer Full on translate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_point_network_full 100 saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train Axial Pointer Linear on translate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_point_network_lines 100 saved_models/translate data/processed/compositional_translate data/results/translate
	@echo "Train MLP on rotate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures mlp_nn 100 saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train CNN on rotate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures cnn 100 saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train Transformer on rotate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures transformer 100 saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train Axial Pointer Full on rotate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_point_network_full 100 saved_models/rotate data/processed/compositional_rotate data/results/rotate
	@echo "Train Axial Pointer Linear on rotate"
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICE) KERAS_BACKEND=$(BACKEND) PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/train_models_main.py --save_figures axial_point_network_lines 100 saved_models/rotate data/processed/compositional_rotate data/results/rotate


## Create Visualisations
visualise_models:
	CUDA_VISIBLE_DEVICES="0" PYTHONPATH=./src KERAS_BACKEND="jax" $(PYTHON_INTERPRETER) src/visualization/visualise_model.py models/experiment/composition/axial_pointer_network.keras ./figures/models data/processed/compositional/test_d2.npz


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

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
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
