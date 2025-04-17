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

/*
Copyright 2016 Fixstars Corporation

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

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <libsgm.h>

#include <time.h>
#include <fstream>
#include "sample_common.h"

static const std::string keys =
"{ @left_img             | <none> | path to input left image                                                            }"
"{ @right_img            | <none> | path to input right image                                                           }"
"{ @left_sparse          | <none> | path to input left sparse disparity                                                 }"
"{ disp_size             |    128 | maximum possible disparity value                                                    }"
"{ P1                    |     10 | penalty on the disparity change by plus or minus 1 between nieghbor pixels          }"
"{ P2                    |    120 | penalty on the disparity change by more than 1 between neighbor pixels              }"
"{ Q1                    |      5 | penalty on the disparity change by plus or minus 1 between nieghbor pixels          }"
"{ Q2                    |    160 | penalty on the disparity change by more than 1 between neighbor pixels              }"
"{ alpha                 |    0.7 | penalty on the disparity change by plus or minus 1 between nieghbor pixels          }"
"{ uniqueness            |   0.95 | margin in ratio by which the best cost function value should be at least second one }"
"{ num_paths             |      8 | number of scanlines used in cost aggregation                                        }"
"{ min_disp              |      0 | minimum disparity value                                                             }"
"{ LR_max_diff           |      1 | maximum allowed difference between left and right disparity                         }"
"{ census_type           |      1 | type of census transform (0:CENSUS_9x7 1:SYMMETRIC_CENSUS_9x7)                      }"
"{ shift8                |      1 | wheather to shift 8 bit for input disparity                                         }"
"{ r_s                   |      6 | window size for semidensification                                                   }"
"{ T_s                   |      2 | threshold for semidensification                                                     }"
"{ r_c                   |     20 | window size for consistency check                                                   }"
"{ T_c                   |      2 | threshold for consistency check                                                     }"
"{ consistency_enable    |      1 | wheather to enable consistency check                                                }"
"{ sd_only               |      0 | 1 for pefroming only semidensification                                              }"
"{ no_sd                 |      0 | 1 for not pefroming semidensification (DSGM)                                        }"
"{ time_log              |  t.txt | log file name for processing time                                                   }"
"{ help h                |        | display this help and exit                                                          }"
"{ output                |    test| output file name.                                                                   }";

int main(int argc, char* argv[])
{
	cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	cv::Mat I1 = cv::imread(parser.get<cv::String>("@left_img"), cv::IMREAD_GRAYSCALE);
	cv::Mat I2 = cv::imread(parser.get<cv::String>("@right_img"), cv::IMREAD_GRAYSCALE);

	cv::Mat S1 = cv::imread(parser.get<cv::String>("@left_sparse"), cv::IMREAD_UNCHANGED);

	const int disp_size = parser.get<int>("disp_size");
	const int P1 = parser.get<int>("P1");
	const int P2 = parser.get<int>("P2");
	const int Pd1 = parser.get<int>("Q1");
	const int Pd2 = parser.get<int>("Q2");
	const float alpha = parser.get<float>("alpha");

	const float uniqueness = parser.get<float>("uniqueness");
	const int num_paths = parser.get<int>("num_paths");
	const int min_disp = parser.get<int>("min_disp");
	const int LR_max_diff = parser.get<int>("LR_max_diff");
	const auto census_type = static_cast<sgm::CensusType>(parser.get<int>("census_type"));

	const bool shift8 = parser.get<bool>("shift8");
	const int spread_size = parser.get<int>("r_s");
	const int spread_threshold = parser.get<int>("T_s");
	const int consistency_size = parser.get<int>("r_c");
	const int consistency_threshold = parser.get<int>("T_c");
	const bool consistency_enable = parser.get<bool>("consistency_enable");

	const bool spread_only = parser.get<bool>("sd_only");
	const bool no_spread = parser.get<bool>("no_sd");

	const std::string output_file = parser.get<std::string>("output");
	const std::string time_log_file = parser.get<std::string>("time_log");

	if (!parser.check()) {
		parser.printErrors();
		parser.printMessage();
		std::exit(EXIT_FAILURE);
	}

	ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed.");
	ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
	ASSERT_MSG(S1.size() == I1.size(), "input sparse depths must be same size as input images.");

	ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(S1.type() == CV_8U || S1.type() == CV_16U, "input sparse depth format must be CV_8U or CV_16U.");

	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scanlines must be 4 or 8.");
	ASSERT_MSG(census_type == sgm::CensusType::CENSUS_9x7 || census_type == sgm::CensusType::SYMMETRIC_CENSUS_9x7, "census type must be 0 or 1.");

	const int src_depth = I1.type() == CV_8U ? 8 : 16;
	const int dst_depth = 16;
	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;

	const sgm::StereoSGM::Parameters param(P1, P2, Pd1, Pd2, alpha, uniqueness, false, path_type, min_disp, LR_max_diff, census_type, shift8, spread_size, spread_threshold, consistency_size, consistency_threshold, consistency_enable);
	sgm::StereoSGM ssgm(I1.cols, I1.rows, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);

	cv::Mat disparity(I1.size(), CV_16U);
	cv::Mat cost(I1.size(), CV_8U);

	clock_t start = clock();
	if (no_spread)
		ssgm.execute(I1.data, I2.data, S1.data, disparity.data);
	else if (spread_only)
		ssgm.execute_spread(I1.data, I2.data, S1.data, disparity.data, cost.data);
	else
		ssgm.execute_spread_sgm(I1.data, I2.data, S1.data, disparity.data);
	clock_t end = clock();

	const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;

	std::ofstream ofs;
	ofs.open(time_log_file, std::ofstream::out | std::ofstream::app);
    ofs << time << std::endl;
	ofs.close();

	// create mask for invalid disp
	const cv::Mat mask = disparity == static_cast<uint16_t>(ssgm.get_invalid_disparity());

	// show image
	cv::Mat disparity_8u, disparity_color;
	disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);

	disparity.setTo(0,mask);
	disparity = disparity * 256;

	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_TURBO);
	disparity_8u.setTo(0, mask);
	disparity_color.setTo(cv::Scalar::all(0), mask);
	if (I1.type() != CV_8U)
		cv::normalize(I1, I1, 0, 255, cv::NORM_MINMAX, CV_8U);
	
	cv::imwrite(output_file+".png", disparity);
	cv::imwrite(output_file+"_color.png", disparity_color);

	return 0;
}
