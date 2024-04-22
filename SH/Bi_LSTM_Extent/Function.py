#!/usr/bin/env python
# coding: utf-8

# In[9]:


def month_mapping(df):
    month_mapping = {
    'Jan': 1, 'Feb': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12}

    df['Month'] = df['Month'].map(month_mapping)
    return(df)


# In[10]:


def date_time(df):
    import pandas as pd
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    
    df.drop(['Year','Month'], axis=1, inplace=True)
    
    return(df)


# In[11]:


def plot_df(df):
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Scatter(
        x=df['Date'],
        y=df['Extent'],
        mode='lines',
        name='Extent'))

    fig.update_traces(hovertemplate='Date: %{x}<br>Extent: %{y}')

    fig.update_layout(
        title='Actual Data',
        xaxis_title='Date',
        yaxis_title='Extent')

    fig.show()
    
def plot_df_with_ma(df):
    import plotly.graph_objects as go
    
    fig = go.Figure(data=go.Scatter(
        x=df['Date'],
        y=df['Extent'],
        mode='lines',
        name='Extent'))

    fig.update_traces(hovertemplate='Date: %{x}<br>Extent: %{y}')

    fig.update_layout(
        title='Actual Data with ma',
        xaxis_title='Date',
        yaxis_title='Area')

    fig.show()


# In[12]:


def df_to_X_y(df, window_size):
    import numpy as np
    
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)


# In[13]:


def training_loss_plot(epochs, loss):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Training Loss', marker='o', linestyle='-')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()


# In[14]:


def train_val_loss_plot(epochs, training_loss, validation_loss):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15,6))
    plt.plot(epochs, training_loss, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, validation_loss, label='Validation Loss', marker='s', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()


# In[15]:


def train_vs_actual_plot(train_results):
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_results.index,
        y=train_results['Train Predictions'],
        mode='lines',
        name='Train Predictions',
        line=dict(color='blue')
        ))
    fig.add_trace(go.Scatter(
        x=train_results.index,
        y=train_results['Actual Extent'],
        mode='lines',
        name='Actual Extent',
        line=dict(color='green')
        ))
    fig.update_layout(
        title='Train Predictions vs Actual Extent',
        xaxis_title='Index',
        yaxis_title='Value'
        )
    fig.show()


# In[ ]:




