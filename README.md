# stronghold-nav
using machine learning to navigate minecraft strongholds

The training pipeline is as follows:
1. load_strongholds.jl loads the strongholds from data/outputDirections.txt
2. reinforcement_environment.jl specifies the reinforcement learning environment
3. reinforcement_recurrent.jl is where training happens
4. model_to_tf.jl prepares models to be saved to tensorflow
5. pickle_to_saved_model.py saves a model as a tensorflow saved model, which can then be used in the StrongholdTrainer mod.