from modules import training_module as tm
model, metrics = tm.train_and_evaluate(model_type="rf", write_predictions=False)
print(metrics['classification_report'])
print(metrics['eval_summary'].head())
