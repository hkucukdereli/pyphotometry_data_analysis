import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Optional
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union, List, Optional

def plot_recording(
    photom,
    signals: Union[str, List[str]],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    figsize: tuple = (12, 6),
    colors: Optional[List[str]] = None,
    alpha: float = 0.8,
    show_events: bool = True,
    event_color: str = 'r',
    event_alpha: float = 0.3,
    show_legend: bool = True,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: str = 'Time (s)',
    dpi: int = 100
) -> tuple:
    """
    Plot photometry recording data for specified signals within a time range.
    
    Args:
        photom: Photom object containing the recording data
        signals: String or list of strings specifying which signals to plot
        start_time: Start time in seconds (default: None, plots from beginning)
        end_time: End time in seconds (default: None, plots until end)
        figsize: Figure size as (width, height) tuple (default: (12, 6))
        colors: List of colors for plotting different signals (default: None)
        alpha: Opacity of the plotted signals (default: 0.8)
        show_events: Whether to show event markers from digital channels (default: True)
        event_color: Color of event markers (default: 'r')
        event_alpha: Opacity of event markers (default: 0.3)
        show_legend: Whether to show legend (default: True)
        title: Plot title (default: None)
        ylabel: Y-axis label (default: None)
        xlabel: X-axis label (default: 'Time (s)')
        dpi: Figure resolution (default: 100)
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Convert single signal to list
    if isinstance(signals, str):
        signals = [signals]
    
    # Set up default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Get time vector in seconds
    time_vector = pd.to_numeric(photom.data_df.index - photom.data_df.index[0]) / pd.Timedelta('1s')
    
    # Set time range
    if start_time is None:
        start_time = time_vector.min()
    if end_time is None:
        end_time = time_vector.max()
    
    # Create mask for time range
    time_mask = (time_vector >= start_time) & (time_vector <= end_time)
    
    # Plot each signal
    for i, signal in enumerate(signals):
        if signal in photom.data_df.columns:
            ax.plot(time_vector[time_mask], 
                   photom.data_df[signal][time_mask],
                   color=colors[i],
                   alpha=alpha,
                   label=signal)
        else:
            print(f"Warning: Signal '{signal}' not found in data")
    
    # Plot events if requested and available
    if show_events:
        for i in [1, 2]:  # Check both digital channels
            pulse_times_key = f'pulse_times_{i}'
            if pulse_times_key in photom.data and photom.data[pulse_times_key] is not None:
                event_times = photom.data[pulse_times_key] / 1000  # Convert from ms to s
                event_mask = (event_times >= start_time) & (event_times <= end_time)
                events = event_times[event_mask]
                
                # Plot vertical lines for events
                for event_time in events:
                    ax.axvline(x=event_time, 
                             color=event_color, 
                             alpha=event_alpha,
                             linestyle='--',
                             zorder=1)
    
    # Customize plot
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    # Set x-axis limits
    ax.set_xlim(start_time, end_time)
    
    # Add legend if requested
    if show_legend:
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def plot_recording777(data_df: pd.DataFrame, 
                   start_time: Optional[str] = None, 
                   end_time: Optional[str] = None,
                   signal_key: str = 'signal_norm'):
    """Plot the recording with optional time windowing."""
    if start_time:
        data_df = data_df[data_df.index >= start_time]
    if end_time:
        data_df = data_df[data_df.index <= end_time]
        
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(data_df.index, data_df[signal_key])
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, ax

# def plot_heatmap(data_df: pd.DataFrame, 
#                  signal_key: str = 'signal_norm', 
#                  bins: int = 24):
#     """Create double-plot heatmap of signal across days."""
#     # Resample data into bins
#     df = data_df[signal_key].resample(f'{24//bins}H').mean()
    
#     # Create matrix for heatmap (days x timepoints)
#     days = (df.index[-1] - df.index[0]).days
#     data_matrix = np.zeros((days, bins))
    
#     for day in range(days):
#         day_data = df[df.index.date == (df.index[0].date() + timedelta(days=day))]
#         data_matrix[day, :len(day_data)] = day_data.values
        
#     # Plot heatmap
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.imshow(data_matrix, cmap='viridis')
#     ax.set_xlabel('Time (hours)')
#     ax.set_ylabel('Days')
#     plt.tight_layout()

#     return fig, ax

def plot_heatmap(data_df: pd.DataFrame, signal_key: str = 'signal_norm', duration: float = 3600) -> plt.Figure:
    """Create double-plot heatmap of signal across days.
    
    Args:
        signal_key: Which signal to plot
        duration: Duration of each bin in seconds (default 3600 = 1 hour)
    """
    # Calculate bins using duration
    seconds_per_day = 24 * 3600  # Seconds in a day
    bins_per_day = int(seconds_per_day / duration)
    
    # Resample data into bins
    df = data_df[signal_key].resample(f'{duration}S').mean()
    
    # Create matrix for heatmap (days x timepoints)
    days = (df.index[-1] - df.index[0]).days
    data_matrix = np.zeros((days, bins_per_day))
    
    for day in range(days):
        day_data = df[df.index.date == (df.index[0].date() + timedelta(days=day))]
        data_matrix[day, :len(day_data)] = day_data.values
            
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(data_matrix, cmap='viridis', aspect='auto')
    
    # # Format x-axis to show hours
    # hours = np.arange(0, 24, duration/3600)
    # ax.set_xticks(np.linspace(0, bins_per_day, len(hours)))
    # ax.set_xticklabels([f'{int(h)}:00' for h in hours])
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Days')
    plt.tight_layout()
    
    return fig, ax

