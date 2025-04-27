import logging
import os

from dotenv import load_dotenv, find_dotenv
from os.path import join
import experiments.analysis.intermediate_layer_analysis_functions as nn_funcs


distance = 0
model_path = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\models\object_compositionality\symmetric_translate_withoutpi'
data_path = r'E:\Projects Large\Learning\Papers_Proposals\2025_Neurips_OOD_Compositionality_Learning\data\object_compositionality\symmetric_translate_withoutpi'
model_filepath = join(model_path, 'axial_pointer_network_full_epoch3600.keras')
data_filepath = join(data_path, f'test_d{distance}.npz')


X, Z, Y = nn_funcs.load_data(data_filepath)

output_layers = nn_funcs.load_models_layers(model_filepath=model_filepath, data_filepath=data_filepath,
                                            selected_layers=['attention_logits', 'spatial_copy_layer'])

flattened_logits = output_layers[0]
spatial_copy_layer = output_layers[1]



batch_size = X.shape[0]
height = X.shape[1]
width = X.shape[2]


visualisations_in, visualisations_out, copied_from_pixel_indices_all_images = \
        nn_funcs.pixel_to_pixel_from_full_attention_logits_matrix_based(input_images=X, attention_logits=flattened_logits,
                                                                        examples_from_batch=list(range(1000)))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
