def get_all_patches_and_positions(position,
                                  patch_size,
                                  overlap,
                                  image_size):
    stride_height = patch_size[0] - overlap[0]
    stride_width = patch_size[1] - overlap[1]

    patch_index = []
    patch_position = []

    for i in range(0, image_size[0] - patch_size[0] + 1, stride_height):
        for j in range(0, image_size[1] - patch_size[1] + 1, stride_width):
            # Calculate patch bounds
            patch_start_row = i
            patch_end_row = i + patch_size[0] - 1
            patch_start_col = j
            patch_end_col = j + patch_size[1] - 1

            # Check if the pixel is within the bounds of the current patch
            if patch_start_row <= position[0] <= patch_end_row and patch_start_col <= position[1] <= patch_end_col:
                # Calculate position within the patch
                position_in_patch_row = position[0] - patch_start_row
                position_in_patch_col = position[1] - patch_start_col
                index = (i // stride_height, j // stride_width)
                patch_index.append(index)
                patch_position.append((
                    position_in_patch_row, position_in_patch_col))

    return patch_index, patch_position
