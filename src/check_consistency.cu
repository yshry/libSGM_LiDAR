/*!
 * libSGM_LiDAR
 * https://github.com/yshry/libSGM_LiDAR
 *
 * Copyright 2025 yshr
 * Released under the MIT license
 * https://github.com/yshry/libSGM_LiDAR/blob/main/LICENSE
 *
 */

#include "internal.h"

#include <cuda_runtime.h>

#include "constants.h"
#include "host_utility.h"

namespace
{

template<typename DST_T>
__global__ void check_consistency_with_sparse_depth_kernel
(bool* invalid_mask, const DST_T* disp, const DST_T* disp_sparse, const uint32_t *occupancy_x, const uint32_t * occupancy_y, 
int width, int height, int num_occupancy, int size,  DST_T threshold=1, unsigned int PROCESS_UNIT = 1u)
{
	const int KERNEL_LENGTH = 2*size + 1;
    const unsigned int KERNEL_AREA = KERNEL_LENGTH * KERNEL_LENGTH;

	const unsigned int k = threadIdx.x * PROCESS_UNIT;
	const unsigned int j = blockIdx.x;

	if (j>= num_occupancy || k >= KERNEL_AREA) return;

	const int x = (int) occupancy_x[j];
	const int y = (int) occupancy_y[j];

	DST_T value_sparse = __ldg(&disp_sparse[y*width + x]);

	for (int i = k; i < k + PROCESS_UNIT; i++)
	{
		if (i>=KERNEL_AREA) continue;

		int dx = i % KERNEL_LENGTH - size;
		int dy = i / KERNEL_LENGTH - size;
		int x_dx = x + dx;
		int y_dy = y + dy;
		if (x_dx<0 || x_dx>= width || y_dy <0 || y_dy>= height) continue;

		const DST_T value = __ldg(&disp[y_dy * width + x_dx]);
		if (abs(value - value_sparse) <= threshold)
			invalid_mask[y_dy * width + x_dx] = false;
	}
}

template<typename SRC_T, typename DST_T>
__global__ void check_consistency_mask_kernel
(bool* invalid_mask, const DST_T* dispL, const DST_T* dispR, const SRC_T* srcL, int width, int height, int src_pitch, int dst_pitch, bool subpixel, int LR_max_diff)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	// left-right consistency check, only on leftDisp, but could be done for rightDisp too

	SRC_T mask = srcL[y * src_pitch + x];
	DST_T org = dispL[y * dst_pitch + x];
	int d = org;
	if (subpixel) {
		d >>= sgm::StereoSGM::SUBPIXEL_SHIFT;
	}
	const int k = x - d;
	if (mask == 0 || org == sgm::INVALID_DISP || (k >= 0 && k < width && LR_max_diff >= 0 && abs(dispR[y * dst_pitch + k] - d) > LR_max_diff)) {
		invalid_mask[y * dst_pitch + x] = true;
	}
}

template<typename DST_T>
__global__ void assign_invalid(DST_T* dispL, const bool* invalid_mask, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	if (invalid_mask[y*width + x]) dispL[y*width + x] = static_cast<DST_T>(sgm::INVALID_DISP);
}

template<typename SRC_T, typename DST_T>
__global__ void check_consistency_kernel(DST_T* dispL, const DST_T* dispR, const SRC_T* srcL, int width, int height, int src_pitch, int dst_pitch, bool subpixel, int LR_max_diff)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;

	// left-right consistency check, only on leftDisp, but could be done for rightDisp too

	SRC_T mask = srcL[y * src_pitch + x];
	DST_T org = dispL[y * dst_pitch + x];
	int d = org;
	if (subpixel) {
		d >>= sgm::StereoSGM::SUBPIXEL_SHIFT;
	}
	const int k = x - d;
	if (mask == 0 || org == sgm::INVALID_DISP || (k >= 0 && k < width && LR_max_diff >= 0 && abs(dispR[y * dst_pitch + k] - d) > LR_max_diff)) {
		// masked or left-right inconsistent pixel -> invalid
		dispL[y * dst_pitch + x] = static_cast<DST_T>(sgm::INVALID_DISP);
	}
}

} // namespace

namespace sgm
{
namespace details
{

void check_consistency(DeviceImage& dispL, const DeviceImage& dispR, const DeviceImage& srcL, bool subpixel, int LR_max_diff)
{
	SGM_ASSERT(dispL.type == SGM_16U && dispR.type == SGM_16U, "");

	const int w = srcL.cols;
	const int h = srcL.rows;

	const dim3 block(16, 16);
	const dim3 grid(divUp(w, block.x), divUp(h, block.y));

	if (srcL.type == SGM_8U) {
		using SRC_T = uint8_t;
		check_consistency_kernel<SRC_T><<<grid, block>>>(dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}
	else if (srcL.type == SGM_16U) {
		using SRC_T = uint16_t;
		check_consistency_kernel<SRC_T><<<grid, block>>>(dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}
	else {
		using SRC_T = uint32_t;
		check_consistency_kernel<SRC_T><<<grid, block>>>(dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}

	CUDA_CHECK(cudaGetLastError());
}

void check_consistency_sparse_disp(DeviceImage& dispL, const DeviceImage& dispR, const DeviceImage& srcL, 
const DeviceImage& dispL_sparse, const DeviceImage& occupancy_x, const DeviceImage& occupancy_y, 
bool subpixel, int LR_max_diff, int size, uint8_t threshold)
{
	SGM_ASSERT(dispL.type == SGM_16U && dispR.type == SGM_16U && dispL_sparse.type == SGM_16U, "");
	SGM_ASSERT(occupancy_x.cols == occupancy_y.cols, "");
	SGM_ASSERT(occupancy_x.rows == occupancy_y.rows, "");

	const int w = srcL.cols;
	const int h = srcL.rows;
	const int occupancy_length = occupancy_x.cols * occupancy_x.rows;

	const dim3 block(16, 16);
	const dim3 grid(divUp(w, block.x), divUp(h, block.y));

	bool* invalid_mask;
	CUDA_CHECK(cudaMalloc((void**)&invalid_mask, w*h*sizeof(bool)));
	CUDA_CHECK(cudaMemset((void*)invalid_mask, 0, w*h*sizeof(bool)));

	if (srcL.type == SGM_8U) {
		using SRC_T = uint8_t;
		check_consistency_mask_kernel<SRC_T><<<grid, block>>>(
			invalid_mask,
			dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}
	else if (srcL.type == SGM_16U) {
		using SRC_T = uint16_t;
		check_consistency_mask_kernel<SRC_T><<<grid, block>>>(
			invalid_mask,
			dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}
	else {
		using SRC_T = uint32_t;
		check_consistency_mask_kernel<SRC_T><<<grid, block>>>(
			invalid_mask,
			dispL.ptr<uint16_t>(), dispR.ptr<uint16_t>(),
			srcL.ptr<SRC_T>(), w, h, srcL.step, dispL.step, subpixel, LR_max_diff);
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	const int KERNEL_LENGTH = 2*size+1;
	const int KERNEL_AREA = KERNEL_LENGTH * KERNEL_LENGTH;

	const unsigned int process_unit = KERNEL_AREA / 1024 + 1;
	const unsigned int bdim_spread = (KERNEL_AREA + process_unit -1)/ process_unit;

	const unsigned int gdim_spread = occupancy_length;

	SGM_ASSERT(bdim_spread <= 1024, "");

	check_consistency_with_sparse_depth_kernel<uint16_t><<<gdim_spread, bdim_spread>>>(invalid_mask, dispL.ptr<uint16_t>(), dispL_sparse.ptr<uint16_t>(), occupancy_x.ptr<uint32_t>(), occupancy_y.ptr<uint32_t>(), w, h, occupancy_length, size, threshold, process_unit);

	CUDA_CHECK(cudaDeviceSynchronize());

	assign_invalid<uint16_t><<<grid, block>>>(dispL.ptr<uint16_t>(), invalid_mask, w, h);

	CUDA_CHECK(cudaFree(invalid_mask));
	CUDA_CHECK(cudaGetLastError());
}

} // namespace details
} // namespace sgm
