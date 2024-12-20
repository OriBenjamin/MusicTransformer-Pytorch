# Piano playing note prediciton - a Music Transformer project

## Table of Contents
### dataset
- EPianoDataset class, which gets a root folder and processes to give x and tgt
- ompute_epiano_accuracy (with softmax)
### dataset_with_hands
- Midi files to learn from (collected by Vista lab)
- Corresponding h5 files for hand positions
### maestro
- The Maestro midi dataset
### dataset_without_hands
- Midi files to learn from (collected by Vista lab)
### model
- The music_transformer model, with generate function for continuing a sample sequence
- SmoothCrossEntropyLoss
### previous_models
- models_maestro: trained parameters based on the Maestro dataset
- start_x: trained parameters based on fine-tuning with dataset_without_hands, starting with epoch x
- saved_model_with_hands: trained parameters based on dataset_with_hands
- finedtuned_model_with_hands: parameters of model trained on the Maestro dataset, and then finetuned (after epoch 70) on dataset_with_hands
- models_maestro
### graph_results
accuracy and loss results of the models trained on dataset_with_hands
### saved_models
- weights and results of last training round
### third_party
- The midi_processor from midi to token sequence (TODO: fix git connection with this folder, remove large files)
### utilities
- argument_funcs: arguments for train, evaluate and generate commands
- run_model: functions for learni=ing a batch and evaluating
### main folder
- evaluate, generate and train
- graph_results
- preprocess_midi: divides a dataset to train, evaluate and test


## About
This project expands an implementation of the MusicTranformer architecture for a project in piano playing prediciton. The original implementation can be found here: https://github.com/gwinndr/MusicTransformer-Pytorch. It is a reproduction of the MusicTransformer (Huang et al., 2018) for Pytorch. This implementation utilizes the generic Transformer implementation introduced in Pytorch 1.2.0 (https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer).

In this project, we explore the application of the musical tranformer architecture and expand its capabilities in representing data, in order to predict musical notes played on a piano, incorporating positional data of the player's hand. Our aim is to enable smart and accessible piano playing, particularly for users requiring assistive technology, by creating a model capable of predicting future notes based on MIDI sequences and hand positioning.

## How to run
1. Download the Maestro dataset (we used v2 but v1 should work as well). You can download the dataset [here](https://magenta.tensorflow.org/datasets/maestro). You only need the MIDI version if you're tight on space. 

2. Run `git submodule update --init --recursive` to get the MIDI pre-processor provided by jason9693 et al. (https://github.com/jason9693/midi-neural-processor), which is used to convert the MIDI file into discrete ordered message types for training and evaluating. 

3. Run `preprocess_midi.py -output_dir <path_to_save_output> <path_to_maestro_data>`, or run with `--help` for details. This will write pre-processed data into folder split into `train`, `val`, and `test` as per Maestro's recommendation.

4. To train a model, run `train.py`. Use `--help` to see the tweakable parameters. See the results section for details on model performance. 

5. After training models, you can evaluate them with `evaluate.py` and generate a MIDI piece with `generate.py`. To graph and compare results visually, use `graph_results.py`.

For the most part, you can just leave most arguments at their default values. If you are using a different dataset location or other such things, you will need to specify that in the arguments. Beyond that, the average user does not have to worry about most of the arguments.

### Training
As an example to train a model using the parameters specified in results:

```
python train.py -output_dir rpr --rpr 
```
You can additonally specify both a weight and print modulus that determine what epochs to save weights and what batches to print. The weights that achieved the best loss and the best accuracy (separate) are always stored in results, regardless of weight modulus input.

### Evaluation
You can evaluate a model using;
```
python evaluate.py -model_weights rpr/results/best_acc_weights.pickle --rpr
```

Your model's results may vary because a random sequence start position is chosen for each evaluation piece. This may be changed in the future.

### Generation
You can generate a piece with a trained model by using:
```
python generate.py -output_dir output -model_weights rpr/results/best_acc_weights.pickle --rpr
```

The default generation method is a sampled probability distribution with the softmaxed output as the weights. You can also use beam search but this simply does not work well and is not recommended.


