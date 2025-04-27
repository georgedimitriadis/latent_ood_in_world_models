.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = latent_ood_in_world_models
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Make Dataset

CYTHON_FILE = optimized_best_first_search.pyx
SETUP_FILE = setup.py
MODULE_NAME = optimized_best_first_search
BUILD_DIR = build
EXTENSION = $(MODULE_NAME).c
SO_FILE = $(MODULE_NAME).so

# Targets
all: $(SO_FILE)

# Rule to build the Cython extension module
cython:
	@echo "Building Cython module..."
	python $(SETUP_FILE) build_ext --inplace

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR) $(EXTENSION) $(SO_FILE)



data_all_experimental:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 1000 data/processed/compositional_translate/train.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":1, "transformation_type": "translate"}'
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_translate/test_d0.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":1, "transformation_type": "translate"}'
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_translate/test_d1.npz generate_compositional_datasets '{"distance":1, "symmetric_objects":1, "transformation_type": "translate"}'
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_translate/test_d2.npz generate_compositional_datasets '{"distance":2, "symmetric_objects":1, "transformation_type": "translate"}'
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 1000 data/processed/compositional_rotate/train.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":0, "transformation_type": "rotate"}'
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_rotate/test_d0.npz generate_compositional_datasets '{"distance":0, "symmetric_objects":0, "transformation_type": "rotate"}'
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_rotate/test_d1.npz generate_compositional_datasets '{"distance":1, "symmetric_objects":0, "transformation_type": "rotate"}'
	CUDA_VISIBLE_DEVICES="" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/experiments/data/generation/generate_datasets_main.py 10 data/processed/compositional_rotate/test_d2.npz generate_compositional_datasets '{"distance":2, "symmetric_objects":0, "transformation_type": "rotate"}'



visualise_model:
	CUDA_VISIBLE_DEVICES="0" PYTHONPATH=./src KERAS_BACKEND="jax" $(PYTHON_INTERPRETER) src/visualization/visualise_model.py models/experiment/composition/axial_pointer_network.keras ./figures/models data/processed/compositional/test_d2.npz



train_cnn:
	CUDA_VISIBLE_DEVICES="1" KERAS_BACKEND="tensorflow" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/cnn_train_generator.py data/processed/cnn_pretrain.npz models/
viz_log:
	 PYTHONPATH=./src $(PYTHON_INTERPRETER) src/visualization/vis.py

cnn_predict:
	KERAS_BACKEND="tensorflow" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/cnn_predict_model.py data/processed/ models/ data/processed/eval/spider dev






train_experiment_composition:
	CUDA_VISIBLE_DEVICES="0" KERAS_BACKEND="jax" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/experiments/train_composition.py models/experiment/composition

train_experiment_task_comp:
	CUDA_VISIBLE_DEVICES="0,1" KERAS_BACKEND="jax" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/experiments/train_task_comp.py models/

text_axial_attention:
	CUDA_VISIBLE_DEVICES="0" KERAS_BACKEND="jax" PYTHONPATH=./src $(PYTHON_INTERPRETER) src/models/test_axial_attention.py


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

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
