/*
Copyright 2025 yshry

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "internal.h"

#include <vector>
#include <cuda_runtime.h>

#include "device_utility.h"
#include "host_utility.h"


namespace sgm
{

namespace spread
{

template <typename T> __device__ inline int popcnt(T x) { return 0; }
template <> __device__ inline int popcnt(uint32_t x) { return __popc(x); }
template <> __device__ inline int popcnt(uint64_t x) { return __popcll(x); }

template<typename COST_TYPE>
__device__ void reduce_min_with_idx
(COST_TYPE* cost_buffer, int* idx_buffer, int reduce_dim)
{
	const int num_threads = blockDim.x;
	const int i = threadIdx.x;
	int step = min(reduce_dim/2, num_threads);
	int num_steps = (reduce_dim + step -1) / step;
	int next_reduce_dim = min(num_threads, reduce_dim/2);

	if (i>=reduce_dim) return;

	for (int n =1; n<num_steps; n++)
	{
		int i_n = i+ n*step;
		if (i_n>=reduce_dim) break;

		if (cost_buffer[i] > cost_buffer[i_n])
		{
			cost_buffer[i] = cost_buffer[i_n];
			idx_buffer[i] = idx_buffer[i_n];
		}
	}
	__syncthreads();

	reduce_min_with_idx(cost_buffer, idx_buffer, next_reduce_dim);
}

template <typename CENSUS_TYPE, typename DISP_TYPE, typename COST_TYPE>
__global__ void spread_disparity_kernel(
	const CENSUS_TYPE *left,
	const CENSUS_TYPE *right,
	const DISP_TYPE *left_sparse,
	const uint32_t *occupancy_x,
	const uint32_t * occupancy_y,
	COST_TYPE *out_cost,
	int width,
	int height,
    int size,
	int num_occupancy,
	unsigned int PROCESS_UNIT = 1u
	)
{
	const int KERNEL_LENGTH = 2*size + 1;
    const unsigned int KERNEL_AREA = KERNEL_LENGTH * KERNEL_LENGTH;
	
	const unsigned int k = threadIdx.x * PROCESS_UNIT;
	const unsigned int j = blockIdx.x;

	if (j>= num_occupancy || k >= KERNEL_AREA) return;

	const int x = (int) occupancy_x[j];
	const int y = (int) occupancy_y[j];

	DISP_TYPE disparity = left_sparse[y*width + x];

	for (int i = k; i < k + PROCESS_UNIT; i++)
	{
		if (i>=KERNEL_AREA) continue;

		int dx = i % KERNEL_LENGTH - size;
		int dy = i / KERNEL_LENGTH - size;
		int x_dx = x + dx;
		int y_dy = y + dy;
		int x_dx_right = x_dx - (int)disparity;
		if (x_dx<0 || x_dx>= width || y_dy <0 || y_dy>= height || x_dx_right <0) continue;

		const CENSUS_TYPE left_value = __ldg(&left[y_dy * width + x_dx]);
		const CENSUS_TYPE right_value = __ldg(&right[y_dy * width + x_dx_right]);
		out_cost[(y_dy*width + x_dx)*KERNEL_AREA + i] = popcnt(left_value^right_value);
	}
}


template <typename DISP_TYPE, typename COST_TYPE, unsigned int MAX_KERNEL_AREA>
__global__ void set_depth_kernel
(
	const DISP_TYPE* input_disparity,
	const COST_TYPE* cost_buffer,
	DISP_TYPE* out_disp,
	int width,
	int height,
	int size,
	COST_TYPE threshold
)
{
	const unsigned int process_unit = (width * height + gridDim.x -1) / gridDim.x; 
	const int kernel_length = 2*size +1;
	const int kernel_area = kernel_length * kernel_length;
	const int num_init = (kernel_area + blockDim.x - 1 )/ blockDim.x;

	const int i0 = blockIdx.x * process_unit;

	__shared__ COST_TYPE reduced_cost[MAX_KERNEL_AREA];
	__shared__ int reduced_idx[MAX_KERNEL_AREA];

	for (int i=0; i<process_unit; i++)
	{
		int pos = i0 + i;
		int x = pos % width;
		int y = pos / width;
		if (y>=height) continue;

		for (int j=0; j<num_init; j++)
		{
			int idx = threadIdx.x + blockDim.x * j;
			if (idx > kernel_area) return;
			reduced_cost[idx] = __ldg(&cost_buffer[(y*width + x)*kernel_area + idx]);
			reduced_idx[idx] = idx;
		}
		__syncthreads();

		reduce_min_with_idx<COST_TYPE>(reduced_cost, reduced_idx, kernel_area);

		if (threadIdx.x == 0)
		{
			int min_idx = reduced_idx[0];
			int dx = -1 * (min_idx % kernel_length - size);
			int dy = -1 * (min_idx / kernel_length - size);
			int x_dx = x + dx;
			int y_dy = y + dy;
			if (reduced_cost[0] < threshold && x_dx>=0 && x_dx < width && y_dy >=0 && y_dy < height)
			{
				out_disp[y*width + x] = __ldg(&input_disparity[y_dy*width + x_dx]);
			}
			else
			{
				out_disp[y*width + x] = __ldg(&input_disparity[y*width + x]);
			}
		}
	}

}

}

namespace details
{

void get_occupancy (const DeviceImage& disp, DeviceImage& occupancy_x, DeviceImage& occupancy_y)
{
	unsigned int width = disp.cols;
	unsigned int height = disp.rows;

	std::vector<uint16_t> disp_host(width*height);

	SGM_ASSERT(disp.type==SGM_8U||disp.type == SGM_16U, "");

	if (disp.type==SGM_8U)
	{
		DeviceImage tmp_disp;
		details::cast_8bit_to_16bit(disp, tmp_disp);
		tmp_disp.download(&disp_host[0]);
	}
	else if (disp.type==SGM_16U)
	{
		disp.download(&disp_host[0]);
	}

	std::vector<uint32_t> vec_occx, vec_occy;
	vec_occx.reserve(width*height);
	vec_occy.reserve(width*height);


	for (unsigned int y=0; y<height; y++)
	{
		for (unsigned int x=0; x<width; x++)
		{
			if (disp_host[y*width+x]>0)
			{
				vec_occy.push_back(y);
				vec_occx.push_back(x);
			}
		}
	}
	occupancy_x.create(1, vec_occx.size(), SGM_32U);
	occupancy_y.create(1, vec_occy.size(), SGM_32U);
	occupancy_x.upload(&vec_occx[0]);
	occupancy_y.upload(&vec_occy[0]);
}	

template <typename CENSUS_TYPE, typename DISP_TYPE, typename COST_TYPE>
void spread_disparity_(const DeviceImage& srcL, const DeviceImage& srcR,
	const DeviceImage& dispL_sparse, const DeviceImage& occupancy_x, const DeviceImage occupancy_y,
	DeviceImage& dst_disp, 
    int size, COST_TYPE threshold)
{
	SGM_ASSERT(occupancy_x.cols == occupancy_y.cols, "");
	SGM_ASSERT(occupancy_x.rows == occupancy_y.rows, "");

	unsigned int width = srcL.cols;
	unsigned int height = srcL.rows;

	unsigned int occupancy_length= occupancy_x.cols * occupancy_x.rows;

	int KERNEL_LENGTH = 2* size + 1;
	int KERNEL_AREA = KERNEL_LENGTH * KERNEL_LENGTH;

	COST_TYPE* cost_buffer_;

	CUDA_CHECK(cudaMalloc((void**)&cost_buffer_, width*height*KERNEL_AREA*sizeof(COST_TYPE)));
	CUDA_CHECK(cudaMemset((void*)cost_buffer_, 0xFF, width*height*KERNEL_AREA*sizeof(COST_TYPE)));

	const unsigned int process_unit = KERNEL_AREA / 1024 + 1;
	const unsigned int bdim_spread = (KERNEL_AREA + process_unit -1)/ process_unit;
	const unsigned int gdim_spread = occupancy_length;
	SGM_ASSERT(bdim_spread <= 1024, "");
	SGM_ASSERT(gdim_spread <= 2^31-1, "");

	const CENSUS_TYPE* left = srcL.ptr<CENSUS_TYPE>();
	const CENSUS_TYPE* right = srcR.ptr<CENSUS_TYPE>();
	const DISP_TYPE* left_sparse = dispL_sparse.ptr<DISP_TYPE>();
	const uint32_t* occ_x = occupancy_x.ptr<uint32_t>();
	const uint32_t* occ_y = occupancy_y.ptr<uint32_t>();

	spread::spread_disparity_kernel<CENSUS_TYPE, DISP_TYPE, COST_TYPE><<<gdim_spread,bdim_spread>>>(left, right, left_sparse, occ_x, occ_y, cost_buffer_, width, height, size, occupancy_length, process_unit);

	DISP_TYPE* out_disp = dst_disp.ptr<DISP_TYPE>();
	const int bdim = WARP_SIZE * 32u;
	const int MAX_RADIUS = 20;
	const int MAX_KERNEL_AREA = (2 * MAX_RADIUS + 1) * (2* MAX_RADIUS +1);
	const unsigned int gdim_reduce = width*height;
	SGM_ASSERT(gdim_reduce<2^31-1, "");
	spread::set_depth_kernel<DISP_TYPE, COST_TYPE, MAX_KERNEL_AREA><<<gdim_reduce,bdim>>>(left_sparse, cost_buffer_, out_disp, width, height, size, threshold);

	CUDA_CHECK(cudaFree(cost_buffer_));
}


void spread_disparity(const DeviceImage& srcL, const DeviceImage& srcR, const DeviceImage& sparseL, const DeviceImage& occupancy_x, const DeviceImage& occupancy_y, DeviceImage& dst_disp, 
	int size, uint8_t threshold)
{
	SGM_ASSERT(srcL.type == srcR.type, "left and right image type must be same.");
	SGM_ASSERT(occupancy_x.type == SGM_32U, "occupnacy_x type must be SGM_32U.")
	SGM_ASSERT(occupancy_y.type == SGM_32U, "occupnacy_y type must be SGM_32U.")
	
	SGM_ASSERT(occupancy_x.rows == occupancy_y.rows, "occupnacy_x and occupancy_y rows must be the same.")
	SGM_ASSERT(occupancy_x.cols == occupancy_y.cols, "occupnacy_x and occupancy_y cols must be the same.")

	dst_disp.create(srcL.rows, srcL.cols, sparseL.type);

	if (srcL.type == SGM_32U)
	{
		if (sparseL.type == SGM_8U)
		{
			spread_disparity_<uint32_t, uint8_t, uint8_t>(srcL, srcR, sparseL, occupancy_x, occupancy_y, dst_disp, size, threshold);
		}
		else if (sparseL.type == SGM_16U)
		{
			spread_disparity_<uint32_t, uint16_t, uint8_t>(srcL, srcR, sparseL, occupancy_x, occupancy_y, dst_disp, size, threshold);
		}
	}
	else if (srcL.type == SGM_64U)
	{
		if (sparseL.type == SGM_8U)
		{
			spread_disparity_<uint64_t, uint8_t, uint8_t>(srcL, srcR, sparseL, occupancy_x, occupancy_y, dst_disp, size, threshold);
		}
		else if (sparseL.type == SGM_16U)
		{
			spread_disparity_<uint64_t, uint16_t, uint8_t>(srcL, srcR, sparseL, occupancy_x, occupancy_y, dst_disp, size, threshold);
		}
	}
}

} // namespace details
} // namespace sgm