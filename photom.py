import numpy as np
from scipy.signal import butter, filtfilt
import json
import os
from typing import Union, List, Dict, Optional
from copy import deepcopy
from scipy.signal import medfilt
from scipy.stats import linregress, zscore
from scipy.optimize import curve_fit
from scipy.signal import decimate

class Photom:
    def __init__(
        self, 
        file_paths: Union[str, List[str]],
        signal_channel: str = "analog_1",
        control_channel: str = "analog_2",
        median_filter: Union[bool, int] = False,
        downsample_factor: Optional[int] = None,
        downsample_method: str = "mean",
        low_pass: float = 10,
        high_pass: float = 0.01,
        normalization: Optional[str] = "dF/F",
        preprocess: bool = True
    ):
        """
        Initialize Photometry analysis object.
        
        Args:
            file_paths: Single file path or list of file paths
            signal_channel: Key for signal channel (default: "analog_1")
            control_channel: Key for control channel (default: "analog_2")
            median_filter: Width of median filter window in samples, False to disable
            downsample_factor: Factor by which to downsample the data (e.g., 2 for half the points)
            downsample_method: Method for downsampling - "mean" or "median"
            low_pass: Low pass filter frequency in Hz
            high_pass: High pass filter frequency in Hz
            normalization: Type of normalization - "dF/F", "z-score", or None
            preprocess: Whether to preprocess the data (default: True)
        """
        assert downsample_method in ["mean", "median"], "downsample_method must be 'mean' or 'median'"
        self.signal_channel = signal_channel
        self.control_channel = control_channel
        self.median_filter = median_filter
        self.downsample_factor = downsample_factor
        self.downsample_method = downsample_method
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.normalization = normalization
        
        # Handle single file path vs list of paths
        if isinstance(file_paths, str):
            self.data, self.metadata = import_ppd(file_paths)
        else:
            # Combine multiple files
            self.data, self.metadata = self._combine_files(file_paths)
        
        if preprocess:
            # Preprocess the data
            self._preprocess()
        else:
            self.processed_data = None
    
    def _combine_files(self, file_paths: List[str]) -> Dict:
        """Combine multiple PPD files into a single dictionary."""
        combined_data = None
        
        for file_path in file_paths:
            current_data = self._import_ppd(file_path)
            
            if combined_data is None:
                combined_data = current_data
                # Store original length for offset calculations
                original_length = len(current_data['time'])
            else:
                # Adjust time array for concatenation
                time_offset = combined_data['time'][-1]
                current_data['time'] += time_offset
                
                # Concatenate arrays
                for key in combined_data.keys():
                    if isinstance(combined_data[key], np.ndarray):
                        combined_data[key] = np.concatenate([combined_data[key], current_data[key]])
                    
                # Adjust pulse times and indices
                if current_data['pulse_inds_1'] is not None:
                    combined_data['pulse_inds_1'] = np.concatenate([
                        combined_data['pulse_inds_1'],
                        current_data['pulse_inds_1'] + original_length
                    ])
                    combined_data['pulse_times_1'] = np.concatenate([
                        combined_data['pulse_times_1'],
                        current_data['pulse_times_1'] + time_offset
                    ])
                
                if current_data.get('pulse_inds_2') is not None:
                    combined_data['pulse_inds_2'] = np.concatenate([
                        combined_data['pulse_inds_2'],
                        current_data['pulse_inds_2'] + original_length
                    ])
                    combined_data['pulse_times_2'] = np.concatenate([
                        combined_data['pulse_times_2'],
                        current_data['pulse_times_2'] + time_offset
                    ])
                
                original_length += len(current_data['time'])
        
        return combined_data
    
    def _apply_filters(self, signals: List[np.ndarray], sampling_rate: float) -> List[np.ndarray]:
        """Apply specified filters to signals."""
        filtered_signals = []
        
        if self.low_pass and self.high_pass:
            b, a = butter(2, np.array([self.high_pass, self.low_pass]) / (0.5 * sampling_rate), "bandpass")
        elif self.low_pass:
            b, a = butter(2, self.low_pass / (0.5 * sampling_rate), "low")
        elif self.high_pass:
            b, a = butter(2, self.high_pass / (0.5 * sampling_rate), "high")
        else:
            return [None] * len(signals)
        
        for signal in signals:
            if signal is not None:
                filtered_signals.append(filtfilt(b, a, signal))
            else:
                filtered_signals.append(None)
        
        return filtered_signals
    
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
        else:  # median
            return np.median(shaped, axis=1)
    
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
    
    def _adjust_digital_events(self, event_times: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Adjust digital event times after downsampling."""
        if self.downsample_factor is None or self.downsample_factor == 1 or event_times is None:
            return event_times
            
        # Convert to new sampling rate
        new_sampling_rate = sampling_rate / self.downsample_factor
        return event_times * sampling_rate / new_sampling_rate

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

    def _preprocess(self):
        """Apply all preprocessing steps to the data."""
        print("Preprocessing data...")
        self.processed_data = deepcopy(self.data)
        
        # Get raw signals
        signal = self.data[self.signal_channel]
        control = self.data[self.control_channel]
        sampling_rate = self.data['sampling_rate']
        
        # 1. Apply median filtering if requested
        if self.median_filter:
            signal = medfilt(signal, self._ensure_odd(self.median_filter))
            control = medfilt(control, self._ensure_odd(self.median_filter))
        
        # 2. Downsample if requested
        if self.downsample_factor and self.downsample_factor > 1:
            # Downsample analog signals
            signal = self._downsample_analog(signal)
            control = self._downsample_analog(control)
            
            # Downsample time
            time = self._downsample_time(self.data['time'])
            
            # Downsample digital signals if present
            for dig_key in ['digital_1', 'digital_2']:
                if dig_key in self.data and self.data[dig_key] is not None:
                    self.processed_data[dig_key] = self._downsample_digital(self.data[dig_key])
            
            # Adjust digital event times
            for times_key in ['pulse_times_1', 'pulse_times_2']:
                if times_key in self.data and self.data[times_key] is not None:
                    self.processed_data[times_key] = self._adjust_digital_events(
                        self.data[times_key],
                        sampling_rate
                    )
            
            # Update time and sampling rate
            self.processed_data['time'] = time
            sampling_rate = sampling_rate / self.downsample_factor
            self.processed_data['sampling_rate'] = sampling_rate
        
        # # 3. Apply low-pass filtering
        # b, a = butter(2, self.low_pass, btype="low", fs=sampling_rate)
        # signal_filt = filtfilt(b, a, signal)
        # control_filt = filtfilt(b, a, control)
        signal_filt = signal
        control_filt = control

        # 4. Photobleaching correction
        t = np.arange(len(signal)) / sampling_rate
        signal_expfit = self._fit_exponential(signal_filt, t, sampling_rate)
        control_expfit = self._fit_exponential(control_filt, t, sampling_rate)
        signal_bc = signal_filt - signal_expfit
        control_bc = control_filt - control_expfit
        
        # 5. Motion correction
        slope, intercept, _, _, _ = linregress(x=control_bc, y=signal_bc)
        est_motion = intercept + slope * control_bc
        signal_mc = signal_bc - est_motion
        
        # 6. Normalization
        if self.normalization == "dF/F":
            signal_norm = 100 * signal_mc / signal_expfit
        elif self.normalization == "z-score":
            signal_norm = zscore(signal_mc)
        else:
            signal_norm = signal_mc
        
        # Store all intermediate and final signals
        self.processed_data.update({
            'signal_filt': signal_filt,
            'signal_bc': signal_bc,
            'signal_mc': signal_mc,
            'signal_norm': signal_norm,
            'control_filt': control_filt,
            'control_bc': control_bc,
            'signal_expfit': signal_expfit,
            'control_expfit': control_expfit,
        })
    
    def get_processed_data(self) -> Dict:
        """Return the processed data dictionary."""
        return self.processed_data

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
    header_dict["filename"]: os.path.basename(file_path)
    
    volts_per_division = header_dict["volts_per_division"]
    sampling_rate = header_dict["sampling_rate"]
    
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
    
    # # Apply filters if specified
    # analog_1_filt, analog_2_filt, analog_3_filt = self._apply_filters(
    #     [analog_1, analog_2, analog_3], sampling_rate
    # )
    
    # Extract digital pulses
    pulse_data = extract_pulses([digital_1, digital_2], sampling_rate)
    
    # Construct output dictionary
    data_dict = {
        "filename": os.path.basename(file_path),
        "start_time": header_dict['start_time'],
        "sampling_rate": sampling_rate,
        "analog_1": analog_1,
        "analog_2": analog_2,
        # "analog_1_filt": analog_1_filt,
        # "analog_2_filt": analog_2_filt,
        "digital_1": digital_1,
        "digital_2": digital_2,
        **pulse_data,
        "time": time,
    }
    
    if n_analog_signals == 3:
        data_dict.update({
            "analog_3": analog_3,
            # "analog_3_filt": analog_3_filt,
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