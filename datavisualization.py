from data_analysis import analyze_data
import plotly.express as px
import matplotlib.pyplot as plt
import torch

def visualize_data():
  train_data = analyze_data()

  # Create bar graph showing leaf data labels
  fig = px.histogram(train_data, x='label')
  fig.update_xaxes(showgrid=False)
  fig.update_yaxes(showgrid=False)
  #fig.show()

  # Show sample of data images
  num_imgs = 15
  fig, axes = plt.subplots(num_imgs//5, 5, figsize=(15,10))
  axes = axes.flatten()
  for i, ax in enumerate(axes):
    idx = torch.randint(len(train_data), size=(1,)).item()
    full_path = train_data.loc[idx]['path']
    ax.imshow(plt.imread(full_path))
    ax.set_title(train_data.loc[idx]['label'])
    ax.set_axis_off()
  #plt.show()

  return train_data

visualize_data()
