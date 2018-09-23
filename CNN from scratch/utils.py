import numpy
import sys

def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = numpy.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                                img.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    # for filter_num in range(conv_filter.shape[0]):
    #     print("Filter ", filter_num + 1)
    #     curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
    #     """ 
    #     Checking if there are mutliple channels for the single filter.
    #     If so, then each channel will convolve the image.
    #     The result of all convolutions are summed to return a single feature map.
    #     """
    #     if len(curr_filter.shape) > 2:
    #         conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
    #         for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
    #             conv_map = conv_map + conv_(img[:, :, ch_num], 
    #                               curr_filter[:, :, ch_num])
    #     else: # There is just a single channel in the filter.
    #         conv_map = conv_(img, curr_filter)
    #     feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.