import os
import math
import pandas as pd
import numpy as np
import sys
from scipy import signal


class CreateFeatureFile:
    def __init__(self, path):
        if path[0] == "/":
            data_path = path
        else:
            data_path = os.getcwd() + path
        print("Loading data from:", data_path)
        self.data = pd.read_csv(data_path, sep=",", header=0, index_col=0)
        # self.data.drop(self.data.tail(1).index,inplace=True)
        print("Done. Loaded data frame (rows, columns):", self.data.shape, "")
        print("")

    def low_pass_filter(self, data):
        fs = 63  # Sampling frequency
        fc = 4  # Cut-off frequency of the filter
        w = fc / (fs / 2)  # Normalize the frequency
        n = 5  # The order of the filter.
        b, a = signal.butter(n, w, 'low')  # Numerator (b) and denominator (a) polynomials of the IIR filter
        return signal.filtfilt(b, a, data)

    def magnitude(self, data):
        result = [0] * len(data["accX"])
        for i in range(1, len(data["accX"])):
            result[i - 1] = math.sqrt(data["accX"][i] * data["accX"][i] + data["accY"][i] * data["accY"][i] +
                                      data["accZ"][i] * data["accZ"][i])
        return result

    def find_peaks_in_signal(self, data):
        mean = self.calculate_mean(data)

        low_pass_magnitude = -(data - mean)
        min = np.amin(low_pass_magnitude)

        low_pass_magnitude = low_pass_magnitude - min
        low_pass_magnitude = low_pass_magnitude * low_pass_magnitude

        min_peak = np.percentile(low_pass_magnitude, 89)
        return signal.find_peaks(low_pass_magnitude, min_peak, distance=25)[0]

    def get_low_pass_magnitude_filter(self, data):
        mag = self.magnitude(data)
        return self.low_pass_filter(mag)

    def create_feature_file_rows(self, username, cycle_range):
        low_pass_magnitude_signal = self.get_low_pass_magnitude_filter(self.data)
        low_pass_magnitude_peaks = self.find_peaks_in_signal(low_pass_magnitude_signal)
        index = 0

        if username[0] == "/":
            writePath = username
        else:
            writePath =  username + "-extracted_features.csv"

        csvFile = open(writePath, "w")
        csvFile.write("avg_len_mag,area_under_cycle,avg_max_in_cycle,avg_min_in_cycle"
                      ",inner_cycle_min_max_diff,avg_max_acc_mag,avg_min_acc_mag,mag_variance,"
                      "avg_max_acc_accX,avg_min_acc_accX,rms_accX,accX_variance,"
                      "avg_max_acc_accY,avg_min_acc_accY,rms_accY,accY_variance,avg_max_acc_accZ,"
                      "avg_min_acc_accZ,rms_accZ\n")
        for i in range(cycle_range, len(low_pass_magnitude_peaks)):
            temp_data = low_pass_magnitude_signal[
                        low_pass_magnitude_peaks[i - cycle_range]: low_pass_magnitude_peaks[i]]

            # fix the issue with the indexes
            temp_data_accX = self.data["accX"][
                             low_pass_magnitude_peaks[i - cycle_range]: low_pass_magnitude_peaks[i]].values
            temp_data_accY = self.data["accY"][
                             low_pass_magnitude_peaks[i - cycle_range]: low_pass_magnitude_peaks[i]].values
            temp_data_accZ = self.data["accZ"][
                             low_pass_magnitude_peaks[i - cycle_range]: low_pass_magnitude_peaks[i]].values

            #csvFile.write(str(index) + ",")
            csvFile.write(str(self.avg_length_calculate(low_pass_magnitude_peaks[i - cycle_range: i])) + ",")
            #csvFile.write(str(self.average_cycle_frequency(low_pass_magnitude_peaks[i - cycle_range: i])) + ",")
            csvFile.write(str(self.area_under_cycle(low_pass_magnitude_signal, low_pass_magnitude_peaks,
                                                    low_pass_magnitude_peaks[i - cycle_range],
                                                    low_pass_magnitude_peaks[i])) + ",")
            csvFile.write(str(self.avg_max_in_cycle(low_pass_magnitude_signal, low_pass_magnitude_peaks,
                                                    low_pass_magnitude_peaks[i - cycle_range],
                                                    low_pass_magnitude_peaks[i])) + ",")
            csvFile.write(str(self.avg_min_in_cycle(low_pass_magnitude_signal, low_pass_magnitude_peaks,
                                                    low_pass_magnitude_peaks[i - cycle_range],
                                                    low_pass_magnitude_peaks[i])) + ",")
            csvFile.write(str(self.inner_cycle_min_max_diff(low_pass_magnitude_signal, low_pass_magnitude_peaks,
                                                            low_pass_magnitude_peaks[i - cycle_range],
                                                            low_pass_magnitude_peaks[i])) + ",")

            csvFile.write(str(self.avg_max_acceleration(temp_data)) + ",")
            csvFile.write(str(self.avg_min_acceleration(temp_data)) + ",")
            csvFile.write(str(self.calculate_variance(temp_data)) + ",")
            #csvFile.write(str(self.calculate_std(temp_data)) + ",")
            #csvFile.write(str(self.calculate_mean(temp_data)) + ",")

            csvFile.write(str(self.avg_max_acceleration(temp_data_accX)) + ",")
            csvFile.write(str(self.avg_min_acceleration(temp_data_accX)) + ",")
            csvFile.write(str(self.root_mean_square(temp_data_accX)) + ",")
            csvFile.write(str(self.calculate_variance(temp_data_accX)) + ",")
            #csvFile.write(str(self.calculate_std(temp_data_accX)) + ",")
            #csvFile.write(str(self.calculate_mean(temp_data_accX)) + ",")

            csvFile.write(str(self.avg_max_acceleration(temp_data_accY)) + ",")
            csvFile.write(str(self.avg_min_acceleration(temp_data_accY)) + ",")
            csvFile.write(str(self.root_mean_square(temp_data_accY)) + ",")
            csvFile.write(str(self.calculate_variance(temp_data_accY)) + ",")
            #csvFile.write(str(self.calculate_std(temp_data_accY)) + ",")
            #csvFile.write(str(self.calculate_mean(temp_data_accY)) + ",")

            csvFile.write(str(self.avg_max_acceleration(temp_data_accZ)) + ",")
            csvFile.write(str(self.avg_min_acceleration(temp_data_accZ)) + ",")
            csvFile.write(str(self.root_mean_square(temp_data_accZ)) + "")
            #csvFile.write(str(self.calculate_variance(temp_data_accZ)) + ",")
            #csvFile.write(str(self.calculate_std(temp_data_accZ)) + ",")
            #csvFile.write(str(self.calculate_mean(temp_data_accZ)))

            csvFile.write("\n")

            index += 1
        csvFile.close()

    # FEATURES
    def avg_length_calculate(self, peaks):
        sum_diff = 0
        for i in range(1, len(peaks)):
            sum_diff += peaks[i] - peaks[i - 1]
        return sum_diff / (len(peaks) - 1)

    def avg_max_acceleration(self, data):
        return np.amax(data)

    def avg_min_acceleration(self, data):
        return np.amin(data)

    # The RMS is an indicator of the gait stability: the higher the RMS value, the lower the degree of stability is
    def root_mean_square(self, data):
        sum_value = 0
        for index in range(1, len(data)):
            sum_value += data[index] * data[index]
        return sum_value / len(data)

    def average_cycle_frequency(self, peaks):
        return 1 / self.avg_length_calculate(peaks)

    def calculate_variance(self, data):
        return np.var(data)

    def calculate_std(self, data):
        return np.std(data)

    def calculate_mean(self, data):
        return np.mean(data)

    def area_under_cycle(self, data, peaks, data_from, data_to):
        area = 0
        num_of_cycles = 0
        for index in range(1, len(peaks)):
            if peaks[index - 1] < data_from:
                continue
            elif peaks[index] > data_to:
                break
            else:
                temp_data = data[peaks[index - 1]: peaks[index]]
                starting_point = temp_data[0]

                for y_value in temp_data:
                    area += y_value - starting_point
                num_of_cycles += 1

        if num_of_cycles == 0:
            return 0
        return area / num_of_cycles

    def avg_max_in_cycle(self, data, peaks, data_from, data_to):
        num_of_max = 0
        num_of_cycle = 0
        for index in range(1, len(peaks)):
            if peaks[index - 1] < data_from:
                continue
            elif peaks[index] > data_to:
                break
            else:
                temp_data = data[peaks[index - 1]: peaks[index]]
                num_of_max += len(signal.find_peaks(temp_data, prominence=0.02)[0])
                num_of_cycle += 1
        if num_of_cycle == 0:
            return 0
        return num_of_max / num_of_cycle

    def avg_min_in_cycle(self, data, peaks, data_from, data_to):
        num_of_max = 0
        num_of_cycle = 0
        for index in range(1, len(peaks)):
            if peaks[index - 1] < data_from:
                continue
            elif peaks[index] > data_to:
                break
            else:
                temp_data = data[peaks[index - 1]: peaks[index]]
                num_of_max += len(signal.find_peaks(-temp_data, prominence=0.02)[0])
                num_of_cycle += 1
        if num_of_cycle == 0:
            return 0
        return num_of_max / num_of_cycle

    def inner_cycle_min_max_diff(self, data, peaks, data_from, data_to):
        mean_diff = 0
        num_of_cycles = 0
        for index in range(1, len(peaks)):
            max_peak = 0
            min_peak = float('Inf')

            if peaks[index - 1] < data_from:
                continue
            elif peaks[index] > data_to:
                break
            else:
                temp_data = data[peaks[index - 1]: peaks[index]]

                for peak in signal.find_peaks(temp_data)[0]:
                    if max_peak < temp_data[peak]:
                        max_peak = temp_data[peak]

                for peak in signal.find_peaks(-temp_data)[0]:
                    if min_peak > temp_data[peak]:
                        min_peak = temp_data[peak]

                if max_peak != 0 and min_peak != float('Inf'):
                    mean_diff += max_peak - min_peak
                num_of_cycles += 1
        if num_of_cycles == 0:
            return 0
        return mean_diff / num_of_cycles


def main(file_name, usernameOrOutPath, cycle_size):
    print("Main parameters: " + str([file_name, usernameOrOutPath, cycle_size]))
    test = CreateFeatureFile(file_name)
    test.create_feature_file_rows(usernameOrOutPath, cycle_size)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("The number parameters is not correct. Expected 3 parameters, got " + str(len(sys.argv)))
        print("Parameters: " + str(sys.argv))
    else:
        print("Parameters: " + str(sys.argv))
        file_name = sys.argv[1]
        username = sys.argv[2]
        cycle_size = int(sys.argv[3])
        main(file_name, username, cycle_size)
