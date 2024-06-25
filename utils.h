#include <vector>
#include <iostream>
#include <stack>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

extern vector<Mat> warped_imgs;
extern vector<vector<Mat>> masks, overlaps, seams;
extern vector<vector<int>> ref_seq;
extern Mat range_map, discovered_time_stamp_map, color_comp_map, range_count_map;

// super parameters
extern double cost_threshold, min_sigma_color, sigma_color, sigma_dist;
extern int propagation_coefficient, height, width;

struct ImgPack
{
public:
    vector<float> PDF;
    vector<float> CDF;
};

namespace Utils
{
    // generate a list consists of points on stitching line.
    vector<Point> build_seam_pixel_lst(Mat& seam_mask);
    Mat refine_seam_pixel_lst_based_abs_RGB_diff_OTSU(vector<Point>& seam_pixel_lst, Mat& warped_ref_img, Mat& warped_tar_img);
    vector<Point> sort_seam_pixel_lst(vector<Point>& seam_pixel_lst, Mat& seam_mask);
    void build_range_map_with_side_addition(Mat& tar_img, vector<Point>& sorted_seam_pixel_lst);
    vector<Point> build_target_pixel_lst(Mat& tar_img);
    void init_color_comp_map(vector<Point>& seam_pixel_lst, Mat& warped_ref_img, Mat& warped_tar_img);

    void update_color_comp_map_range_anomaly(vector<Point>& pxl_lst, Mat& warped_tar_img, vector<Point>& sorted_seam_pixel_lst, Mat& anomaly_mask);
    vector<double> get_color_comp_value_range_anomaly(Point& p, Mat& warped_tar_img, vector<Point>& sorted_seam_pixel_lst, Mat& anomaly_mask);

    void single_img_correction(int i, const vector<bool>& is_corrected);

    //Fecker func
    ImgPack CalDF(Mat& src, int channel, Mat& overlap);
    Mat HM_Fecker(Mat& ref, Mat& tar, Mat& overlap);

    //ordering algorithm
    int getNextImageID(const vector<bool>& is_corrected);

    //Proposed fusion color correction method
    vector<int> MST();

    //hole fulling
    Mat hole_detection(Mat& result_img);
    void hole_filling(Mat& result_img, Mat& hole);

    // build result
    void getFeckerWeightingCorrectedImg(Mat& warped_tar_img, int num, vector<Mat>& cites_range, vector<Mat>& wave_num,
                                        vector<Mat>& comps, vector<Mat>& Fecker_comps, vector<double>& seam_lengths);
    Mat build_final_result();
}

namespace Inits
{
    // load data.
    vector<string> getImgFilenameList(string& addr);
    vector<Mat> LoadImage(string& addr, int flag);
    vector<vector<Mat>> LoadMask(string& addr, int N);
    void loadAll(string& warpDir, string& seamDir, string& overlapDir, string& maskDir);

    //initialization
    void Initialize_var();
}