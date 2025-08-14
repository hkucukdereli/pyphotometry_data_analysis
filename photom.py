import os
import json
from tqdm import tqdm
from typing import Union, List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import medfilt, butter, filtfilt, decimate
from scipy.stats import linregress, zscore
from scipy.optimize import curve_fit

class Photom:
    def __init__(
        self, 
        file_paths: Union[str, List[str]],
        platform: str = "pyphotometry",
        signal_channel: str = "analog_1",
        control_channel: str = "analog_2",
        low_pass: float = None,
        high_pass: float = None,
        preprocess: bool = False,
        median_filter: Union[bool, int] = False,
        downsample_factor: Optional[int] = None,
        downsample_method: str = "mean",
        bleach_correct: bool = False,
        motion_correct: bool = True,
        normalization: Optional[str] = "dF/F",
        timeseries: bool = True,
        smooth_timeseries: bool = False,
        as_df: bool = True,
        verbose: bool = True
    ):
        """
        Initialize Photometry analysis object.
        
        Args:
            file_paths: Single file path or list of file paths
            platform: Recording platform (default: "pyphotometry")
            signal_channel: Key for signal channel (default: "analog_1")
            control_channel: Key for control channel (default: "analog_2")
            low_pass: Low pass filter frequency in Hz
            high_pass: High pass filter frequency in Hz
            median_filter: Width of median filter window in samples, False to disable
            downsample_factor: Factor by which to temporally downsample the data
            downsample_method: Method for downsampling - "mean" or "median"
            bleach_correct: Whether to apply photobleaching correction (default: False)
            motion_correct: Whether to apply motion correction (default: True)
            normalization: Type of normalization - "dF/F", "z-score", "zscore", or None
            preprocess: Whether to preprocess the data (default: True). If False, only raw and filtered data is stored.
            timeseries: Whether to return time series data (default: True)
            smooth_timeseries: Whether to smooth timestamps to the nearest sampling interval (default: False)
            as_df: Whether to return data as a DataFrame (default: True)
            verbose: Whether to print progress messages (default: True)
        """
        assert downsample_method in ["mean", "median"], "downsample_method must be 'mean' or 'median'"
        self.platform = platform
        self.signal_channel = signal_channel
        self.control_channel = control_channel
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.median_filter = median_filter
        self.downsample_factor = downsample_factor
        self.downsample_method = downsample_method
        self.bleach_correct = bleach_correct
        self.motion_correct = motion_correct
        self.normalization = normalization
        self.verbose = verbose
        
        # Initialize processing history
        self.__init_processing()

        # Load data
        # Handle single file path vs list of paths
        if isinstance(file_paths, str):
            self.data, self.metadata = self.import_data(file_paths)
            self.sampling_rate = self.metadata['sampling_rate']
            # Update processing history
            self.processing_history.append({
                'order': self._add_step(),
                'step': 'data loaded',
                'multiple files': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            # Combine multiple files
            self.data, self.metadata = self._combine_files(file_paths)
            self.sampling_rate = self.metadata['sampling_rate']
            # Update processing history
            self.processing_history.append({
                'order': self._add_step(),
                'step': 'data loaded',
                'multiple files': True,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        self.data['signal'] = self.data[signal_channel]
        self.data['control'] = self.data[control_channel]

        # Apply low-pass filtering
        (self.data['signal_filt'], 
         self.data['control_filt']) = self.apply_filters(['signal', 'control'], 
                                                         sampling_rate = self.sampling_rate, 
                                                         low_pass = self.low_pass, 
                                                         high_pass = self.high_pass
                                                         )
        
        if preprocess:
            # Preprocess the data
            self.preprocess()
            if self.verbose: print("Data preprocessed. See processing_history for details.")
            self.preprocessed = True
        else:
            self.preprocessed = False

        if timeseries:
            # Add timestamps to data
            self._add_timeseries(smooth_timeseries=smooth_timeseries)
            
            if as_df:
                if self.verbose: print("Contructing the DataFrame...")
                df_dict = {key:self.data[key] for key in self.data.keys() if len(self.data[key])==len(self.data['time'])}
                self.data_df = pd.DataFrame(df_dict)
                self.data_df = self.data_df.set_index('timestamps', drop=True)
                self.data_df.index.name = None
                # self.data_df.drop(columns=['time'], inplace=True)
                self.data_df = self.data_df.sort_index()
                if self.verbose: print("DataFrame is ready and accessible as Photom.data_df")
        else:
            if as_df:
                if self.verbose: print("Contructing the DataFrame...")
                df_dict = {key:self.data[key] for key in self.data.keys() if len(self.data[key])==len(self.data['time'])}
                self.data_df = pd.DataFrame(df_dict)
                if self.verbose: print("DataFrame is ready and accessible as Photom.data_df")
    
    # File IO methods
    def import_data(self, file_path: str):
        """Return the appropriate import function based on file type and platform."""
        if file_path.endswith(".ppd"):
            self.platform = "pyphotometry"  # Overwrite the platform if it's a ppd file
            return import_ppd
        elif file_path.endswith(".csv") and self.platform == "pyphotometry":
            return import_csv_pyphotometry
        elif file_path.endswith(".csv") and self.platform == "neurophotometrics":
            return import_csv_neurophotometrics
        else:
            raise ValueError("Unsupported file type or platform!")

    def _combine_files(self, file_paths: List[str],
                       allow_mixed_modes: bool = False, 
                       allow_mixed_subjects: bool = False) -> Dict:
        
        metadata_temp = import_metadata(file_paths[0])  # Load metadata from first file

        Data, Metadata = [], []
        for i, file_path in tqdm(enumerate(file_paths), total=len(file_paths), desc="Loading files..."):
            data, metadata = import_ppd(file_path)
            metadata['session_id'] = i + 1  # Assign session ID
            metadata['session_len'] = len(data['time'])
            
            # Ensure consistency across recordings
            if metadata_temp['sampling_rate'] != metadata['sampling_rate']:
                raise ValueError("Concatenating recordings with different sampling rates is not allowed. "
                                "Load them individually and downsample to the same rate before concatenation.")
            if not allow_mixed_subjects and metadata_temp['subject_ID'] != metadata['subject_ID']:
                raise ValueError("You are trying to concatenate recordings from multiple subjects. "
                                "If intentional, set allow_mixed_subjects=True.")
            if not allow_mixed_modes and metadata_temp['mode'] != metadata['mode']:
                raise ValueError("You are trying to concatenate recordings with different modes. "
                                "If intentional, set allow_mixed_modes=True.")

            Data.append(data)
            Metadata.append(metadata)

        # Combine metadata
        combined_metadata = {key: [d[key] for d in Metadata] for key in Metadata[0]}
        combined_metadata['version'] = np.unique(combined_metadata['version'])[0]
        combined_metadata['sampling_rate'] = np.unique(combined_metadata['sampling_rate'])[0]

        if not allow_mixed_subjects:
            combined_metadata['subject_ID'] = np.unique(combined_metadata['subject_ID'])[0]
        if not allow_mixed_modes:
            combined_metadata['mode'] = np.unique(combined_metadata['mode'])[0]

        # Find shared keys across all data dictionaries
        shared_keys = set(Data[0].keys()).intersection(*(d.keys() for d in Data[1:]))

        # Combine data using shared keys
        combined_data = {key: np.concatenate([d[key] for d in Data if key in d]) for key in shared_keys}

        return combined_data, combined_metadata
    
    # Processing history methods
    def __init_processing(self):
        # Initialize processing history
        self.processing_history = []
        self.__processing_order = 0

    def _add_step(self):
        self.__processing_order += 1
        return self.__processing_order

    def save_processing_history(self, savepath=None):
        """
        Save the processing history to a JSON file.
        
        Parameters:
        -----------
        filename : str, optional
            The filename or directory to save the history to. 
            - If None, saves to current script directory with timestamp filename
            - If directory path, saves in that directory with timestamp filename
            - If file path, saves to that specific file
        
        Returns:
        --------
        str
            The path to the saved JSON file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"processing_history_{timestamp}.json"
        
        if savepath is None:
            # Save in current script directory
            savepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), default_filename)
        elif os.path.isdir(savepath):
            # If filename is a directory, save file in that directory
            savepath = os.path.join(savepath, default_filename)

        # Ensure the directory exists
        directory = os.path.dirname(savepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Write the processing history to the JSON file
        with open(savepath, 'w') as f:
            json.dump(self.processing_history, f, indent=2)
        
        return savepath

    def _add_timeseries(self, smooth_timeseries: bool = False):
        # Combine time arrays monotically
        if not 'session_id' in self.metadata:
            start_time = pd.to_datetime(self.metadata['start_time'])
            if smooth_timeseries:
                start_time = smooth_timestamp(start_time, 1000 / self.sampling_rate)

            start_time_list = np.repeat(pd.to_datetime(start_time), len(self.data['time']))

            time_deltas = np.array([timedelta(milliseconds=t) for t in self.data['time']])
            self.data['timestamps'] = start_time_list + time_deltas
        else:
            # Create timeseries for each recording session using the start time for each recording
            session_inds = np.cumsum(np.concatenate([[0], self.metadata['session_len']]))
            self.data['timestamps'] = []
            for i, session_id in enumerate(self.metadata['session_id']):
                start_time = pd.to_datetime(self.metadata['start_time'][i])
                if smooth_timeseries:
                    start_time = smooth_timestamp(start_time, 1000 / self.sampling_rate)

                time = self.data['time'][session_inds[i]:session_inds[i+1]]
                start_time_list = np.repeat(pd.to_datetime(start_time), len(time))

                time_deltas = np.array([timedelta(milliseconds=t) for t in time])
                self.data['timestamps'].append(start_time_list + time_deltas)
            self.data['timestamps'] = np.concatenate(self.data['timestamps'])

    def apply_filters(self, 
                      signals: Union[str, List[str]], 
                      sampling_rate: float = None, 
                      low_pass: float = None, 
                      high_pass: float = None,
                      median: Union[bool, int] = False) -> List[np.ndarray]:
        """Apply specified filters to signals."""
        # Convert single string to list
        if isinstance(signals, str):
            signals = [signals]

        sampling_rate = sampling_rate or self.sampling_rate
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be specified.")

        filtered_signals = []
            
        if low_pass and high_pass:
            b, a = butter(2, np.array([high_pass, low_pass]) / (0.5 * sampling_rate), "bandpass")
        elif low_pass:
            b, a = butter(2, low_pass / (0.5 * sampling_rate), "low")
        elif high_pass:
            b, a = butter(2, high_pass / (0.5 * sampling_rate), "high")
        else:
            if median:
                filtered_signals = [medfilt(self.data[sig], kernel_size=self._ensure_odd(median)) if self.data[sig] is not None else None for sig in signals]
                return filtered_signals[0] if len(signals)==1 else filtered_signals
            else:
                if self.verbose: print("No filter specified. Returning unfiltered signals.")
                unfiltered = [self.data[sig] for sig in signals]
                return unfiltered[0] if len(signals)==1 else unfiltered
            
        for signal_name in signals:
            if signal_name in self.data and self.data[signal_name] is not None:
                filtered_signals.append(filtfilt(b, a, self.data[signal_name]))
            else:
                filtered_signals.append(None)

        # Update processing history
        self.processing_history.append({
            'order': self._add_step(),
            'step': 'filtering',
            'low pass': low_pass,
            'high pass': high_pass,
            'median': median,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return filtered_signals[0] if len(signals)==1 else filtered_signals
    
    def _downsample_analog(self, signal: np.ndarray) -> np.ndarray:
        """Downsample analog signal using specified method."""
        if self.downsample_factor is None or self.downsample_factor == 1:
            return signal
            
        # Truncate signal to make it divisible by downsample_factor
        trunc_length = len(signal) - (len(signal) % self.downsample_factor)
        signal = signal[:trunc_length]
        
        # Reshape to prepare for downsampling
        shaped = signal.reshape(-1, self.downsample_factor)
        
        # Apply downsampling method
        if self.downsample_method == "mean":
            return np.mean(shaped, axis=1)
        elif self.downsample_method == "median":
            return np.median(shaped, axis=1)
        else:
            raise ValueError("Invalid downsampling method. Must be 'mean' or 'median'.")
    
    def _downsample_digital(self, digital: np.ndarray) -> np.ndarray:
        """Downsample digital signal by preserving any HIGH values in each bin."""
        if self.downsample_factor is None or self.downsample_factor == 1:
            return digital
            
        # Truncate signal
        trunc_length = len(digital) - (len(digital) % self.downsample_factor)
        digital = digital[:trunc_length]
        
        # Reshape and use any HIGH value in bin
        shaped = digital.reshape(-1, self.downsample_factor)
        return np.any(shaped, axis=1).astype(int)
    
    def _downsample_time(self, time: np.ndarray) -> np.ndarray:
        """Downsample time array by taking first value of each bin."""
        if self.downsample_factor is None or self.downsample_factor == 1:
            return time
            
        # Truncate signal
        trunc_length = len(time) - (len(time) % self.downsample_factor)
        time = time[:trunc_length]
        
        # Take first time point of each bin
        return time[::self.downsample_factor]
    
    def _adjust_digital_events(self, event_times: np.ndarray) -> np.ndarray:
        """Adjust digital event times after downsampling."""
        if self.downsample_factor is None or self.downsample_factor == 1 or event_times is None:
            return event_times
            
        return event_times / self.downsample_factor

    @staticmethod
    def _double_exponential(t, const, amp_fast, amp_slow, tau_fast, tau_slow):
        """Compute double exponential function with constant offset."""
        return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)

    def _fit_exponential(self, signal, t, sampling_rate):
        """Fit double exponential to signal after downsampling to 1Hz."""
        max_sig = np.max(signal)
        signal_ds = decimate(signal, int(sampling_rate))
        t_ds = decimate(t, int(sampling_rate))
        
        # Initial parameters and bounds for fitting
        init_params = [max_sig / 2, max_sig / 4, max_sig / 4, 300, 3000]
        bounds = ([0, 0, 0, 60, 600], [max_sig, max_sig, max_sig, 600, 36000])
        
        params, _ = curve_fit(
            self._double_exponential, 
            t_ds, 
            signal_ds, 
            p0=init_params, 
            bounds=bounds, 
            maxfev=1000
        )
        
        return self._double_exponential(t, *params)

    @staticmethod
    def _ensure_odd(n):
        """Ensure number is odd by adding 1 if necessary."""
        return n if n % 2 == 1 else n + 1

    # Method to preprocess the data
    def preprocess(self):
        """Apply all preprocessing steps to the data.

        Preprocessing steps:
        1. Median filtering
        2. Downsampling
        3. Photobleaching correction
        4. Motion correction
        5. Normalization
        """
        if self.verbose: print("Preprocessing data...")

        # Get filtered signals if available
        signal = self.data[f'{self.signal_channel}_filt'] if f'{self.signal_channel}_filt' in self.data else self.data[self.signal_channel]
        control = self.data[f'{self.control_channel}_filt'] if f'{self.control_channel}_filt' in self.data else self.data[self.control_channel]
        self.data.pop(self.signal_channel)
        self.data.pop(self.control_channel)

        # Track the latest version of signals
        current_signal = signal
        current_control = control
        current_sampling_rate = self.sampling_rate
        current_time = self.data['time']

        # 1. Apply median filtering if needed
        if self.median_filter:
            current_signal = medfilt(current_signal, self._ensure_odd(self.median_filter))
            current_control = medfilt(current_control, self._ensure_odd(self.median_filter))

            # Add filtered signals to data
            self.data.update({
                'signal_med': current_signal,
                'control_med': current_control
            })

            # Update processing history
            self.processing_history.append({
                'order': self._add_step(),
                'step': 'median filtering',
                'kernel_size': self.median_filter,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # 2. Downsample if needed
        if self.downsample_factor and self.downsample_factor > 1:
            # Downsample analog signals
            current_signal = self._downsample_analog(current_signal)
            current_control = self._downsample_analog(current_control)
            
            # Downsample time
            current_time = self._downsample_time(self.data['time'])

            old_rate = current_sampling_rate
            current_sampling_rate = current_sampling_rate / self.downsample_factor
            self.sampling_rate = current_sampling_rate
            
            # Add downsampled signals to data
            self.data.update({'time': current_time})
            self.metadata.update({
                'original_sampling_rate': old_rate,
                'sampling_rate': current_sampling_rate
            })

            # Update processing history
            self.processing_history.append({
                'order': self._add_step(),
                'step': 'downsample',
                'factor': self.downsample_factor,
                'original_sampling_rate': old_rate,
                'final_sampling_rate': current_sampling_rate,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Downsample digital signals if present
            for dig_key in ['digital_1', 'digital_2']:
                if dig_key in self.data and self.data[dig_key] is not None:
                    self.data[dig_key] = self._downsample_digital(self.data[dig_key])
            
            # Adjust digital event times
            for times_key in ['pulse_times_1', 'pulse_times_2']:
                if times_key in self.data and self.data[times_key] is not None:
                    self.data[times_key] = self._adjust_digital_events(self.data[times_key])
            
            # Update time and sampling rate
            self.data['time'] = current_time
            self.sampling_rate = self.sampling_rate / self.downsample_factor
            self.metadata['sampling_rate'] = self.sampling_rate

        # 3. Photobleaching correction if needed
        if self.bleach_correct:
            t = np.arange(len(current_signal)) / current_sampling_rate
            signal_expfit = self._fit_exponential(current_signal, t, current_sampling_rate)
            control_expfit = self._fit_exponential(current_control, t, current_sampling_rate)
            
            current_signal = current_signal - signal_expfit
            current_control = current_control - control_expfit

            # Add bleach-corrected signals to data
            self.data.update({
                'signal_bc': current_signal,
                'control_bc': current_control,
                'signal_expfit': signal_expfit,
                'control_expfit': control_expfit
            })

            # Update processing history
            self.processing_history.append({
                'order': self._add_step(),
                'step': 'bleach correction',
                'method': 'exponential fit',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        # 4. Motion correction if needed
        if self.motion_correct:
            slope, intercept, _, _, _ = linregress(x=current_control, y=current_signal)
            est_motion = intercept + slope * current_control
            current_signal = current_signal - est_motion
            
            # Add motion-corrected signal to data
            self.data.update({'signal_mc': current_signal})

            # Update processing history
            self.processing_history.append({
                'order': self._add_step(),
                'step': 'motion correction',
                'method': 'linear regression',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # 5. Normalization
        if self.normalization == "dF/F":
            # Calculate the baseline
            if 'signal_expfit' in self.data: # Check if we have expfit from bleach correction
                signal_baseline = self.data['signal_expfit']

            signal_norm = 100 * current_signal / signal_baseline
            self.data.update({'signal_norm': signal_norm})
        elif self.normalization == "z-score" or self.normalization == "zscore":
            signal_norm = zscore(current_signal)
            self.data.update({'signal_norm': signal_norm})

        # Update processing history
        self.processing_history.append({
            'order': self._add_step(),
            'step': 'normalization',
            'method': self.normalization,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
####################
# Helper functions #
####################

def import_metadata(file_path: str) -> Dict:
    """
    Import metadata from PPD file.

    Args:
        file_path: Path to PPD file

    Returns:
        header_dict: Dictionary containing header information
    """
    with open(file_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        data_header = f.read(header_size)
        
    # Extract header information
    header_dict = json.loads(data_header)
    header_dict["filename"] = os.path.basename(file_path)
    
    if 'date_time' in header_dict:
        header_dict['start_time'] = header_dict['date_time']
        del header_dict['date_time']

    return header_dict

def import_ppd(file_path: str) -> Dict:
    """
    Import single PPD file.

    Args:
        file_path: Path to PPD file

    Returns:
        data_dict: Dictionary containing data from PPD file
        header_dict: Dictionary containing header information
    """
    with open(file_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype("<u2"))
        
    # Extract header information
    header_dict = json.loads(data_header)
    header_dict["filename"] = os.path.basename(file_path)
    
    volts_per_division = header_dict["volts_per_division"]
    sampling_rate = header_dict["sampling_rate"]
    
    if 'date_time' in header_dict:
        header_dict['start_time'] = header_dict['date_time']
        del header_dict['date_time']

    # Extract signals
    analog = data >> 1
    digital = ((data & 1) == 1).astype(int)
    
    # Get number of signals
    if "n_analog_signals" in header_dict:
        n_analog_signals = header_dict["n_analog_signals"]
        n_digital_signals = header_dict["n_digital_signals"]
    else:
        n_analog_signals = 2
        n_digital_signals = 2
        
    # Extract individual signals
    analog_1 = analog[::n_analog_signals] * volts_per_division[0]
    analog_2 = analog[1::n_analog_signals] * volts_per_division[1]
    analog_3 = analog[2::n_analog_signals] * volts_per_division[0] if n_analog_signals == 3 else None
    digital_1 = digital[::n_analog_signals]
    digital_2 = digital[1::n_analog_signals] if n_digital_signals == 2 else None
    
    # Calculate time array
    time = np.arange(analog_1.shape[0]) * 1000 / sampling_rate
    
    # Extract digital pulses
    pulse_data = extract_pulses([digital_1, digital_2], sampling_rate)
    
    # Construct output dictionary
    data_dict = {
        "analog_1": analog_1,
        "analog_2": analog_2,
        "digital_1": digital_1,
        "digital_2": digital_2,
        **pulse_data,
        "time": time,
    }
    
    if n_analog_signals == 3:
        data_dict.update({
            "analog_3": analog_3
        })
        
    return data_dict, header_dict

def extract_pulses(digital_signals: List[np.ndarray], sampling_rate: float) -> Dict:
    """
    Extract pulse times and indices from digital signals.
    
    Args:
        digital_signals: List of digital signals
        sampling_rate: Sampling rate of the signals
        
    Returns:
        pulse_data: Dictionary containing pulse indices and times
    """
    pulse_data = {}
    
    for i, signal in enumerate(digital_signals, 1):
        if signal is not None:
            pulse_inds = 1 + np.where(np.diff(signal) == 1)[0]
            pulse_times = pulse_inds * 1000 / sampling_rate
        else:
            pulse_inds = None
            pulse_times = None
            
        pulse_data[f'pulse_inds_{i}'] = pulse_inds
        pulse_data[f'pulse_times_{i}'] = pulse_times
    
    return pulse_data

def add_zt_columns(df, zt0_hour=18, fill_gaps=False, smooth=True):
    """
    Add ZT hour and experimental day columns, with Day 1 starting at first ZT0.
    Anything before first ZT0 is Day 0.
    
    Args:
        df: DataFrame with a datetime index
        zt0_hour: Hour of the day when ZT0 starts (default: 18 for 18:00)
        fill_gaps: If True, fills gaps in the time series based on estimated sampling rate
                   while preserving all original data points
        
    Returns:
        DataFrame with added ZT and Day columns
    """
    df_with_zt = df.copy()
    
    # If fill_gaps is True, estimate sampling rate and fill gaps
    if fill_gaps:
        # Sort the index to ensure timestamps are in order
        df_with_zt = df_with_zt.sort_index()
        
        # Calculate time differences between consecutive samples (in milliseconds)
        time_diffs = df_with_zt.index.to_series().diff().dt.total_seconds() * 1000
        
        # Remove NaN (first row) and find the most common difference using the first 1000 samples
        # or fewer if the dataframe is smaller
        sample_size = min(1000, len(time_diffs) - 1)
        common_diffs = time_diffs.iloc[1:sample_size+1]
        sampling_rate_ms = int(round(common_diffs.median()))
        
        print(f"Estimated sampling rate: {1000/sampling_rate_ms}Hz")
        
        # Create a new index with regular intervals
        start_time = df_with_zt.index.min()
        end_time = df_with_zt.index.max()
        complete_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{sampling_rate_ms}ms'
        )
        
        # Create a new dataframe with the complete index
        df_complete = pd.DataFrame(index=complete_index)
        
        # Merge with original data, keeping all original values
        df_with_zt = df_with_zt.join(df_complete, how='outer')
        
        # Sort by timestamp again to ensure proper order
        df_with_zt = df_with_zt.sort_index()
    
    # Calculate ZT hours
    df_with_zt['ZT'] = (df_with_zt.index.hour - zt0_hour) % 24
    
    # Find first ZT0 timestamp using only non-NaN rows for original data
    # First get a mask of rows that were in the original dataset (any non-NaN value)
    original_rows = ~df_with_zt.iloc[:, 0].isna() if len(df_with_zt.columns) > 0 else pd.Series(True, index=df_with_zt.index)
    
    # Find the first ZT0 among original rows
    first_zt0_mask = (df_with_zt.index.hour == zt0_hour) & original_rows
    
    if not any(first_zt0_mask):
        raise ValueError("No ZT0 found in data")
    
    first_zt0 = df_with_zt.index[first_zt0_mask][0]
    
    # Calculate days from first ZT0
    days_from_zt0 = (df_with_zt.index.date - first_zt0.date()).astype('timedelta64[D]').astype(int)
    
    # Initialize teh day count
    df_with_zt['day'] = 0
    
    # Set Day 1 and onwards
    after_first_zt0_mask = (df_with_zt.index >= first_zt0)
    df_with_zt.loc[after_first_zt0_mask, 'day'] = days_from_zt0[after_first_zt0_mask] + 1
    
    # Adjust for hours before ZT0 within each day
    day_adjustment_mask = (df_with_zt.index > first_zt0) & (df_with_zt.index.hour < zt0_hour)
    df_with_zt.loc[day_adjustment_mask, 'day'] -= 1
    
    return df_with_zt

def smooth_timestamp(timestamp, sampling_rate_ms=None):
    """
    Smooths a single timestamp to the closest interval based on the sampling rate.
    
    Args:
        timestamp: The timestamp to smooth (pandas Timestamp or compatible datetime)
        sampling_rate_ms: Sampling rate in milliseconds (default: 100ms)
    
    Returns:
        pandas.Timestamp: The smoothed timestamp
    """
    # Convert timestamp to total microseconds
    seconds = timestamp.second
    microseconds = timestamp.microsecond
    total_micros = seconds * 1_000_000 + microseconds
    
    # Convert sampling rate to microseconds
    sampling_micros = sampling_rate_ms * 1000
    
    # Round to nearest sampling interval
    rounded_micros = round(total_micros / sampling_micros) * sampling_micros
    
    # Convert back to seconds and microseconds
    new_seconds = int(rounded_micros // 1_000_000)
    new_microseconds = int(rounded_micros % 1_000_000)
    
    # Create new timestamp
    smoothed_timestamp = pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=timestamp.day,
        hour=timestamp.hour,
        minute=timestamp.minute,
        second=new_seconds,
        microsecond=new_microseconds
    )
    
    return smoothed_timestamp
