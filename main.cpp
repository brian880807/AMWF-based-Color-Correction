#include "utils.h"

vector<Mat> warped_imgs;
vector<vector<Mat>> masks, overlaps, seams;
vector<vector<int>> ref_seq;
Mat range_map, discovered_time_stamp_map, color_comp_map, range_count_map;

// parameters
double cost_threshold, min_sigma_color, sigma_color, sigma_dist;
int propagation_coefficient, height, width;

int main()
{
    // parameters setting
    sigma_dist = 10;
    sigma_color = 10;
    min_sigma_color = 0.5;
    cost_threshold = 500;
    propagation_coefficient = 2;

    string data_path = "../datasets/01/";

    string warpDir = data_path + "multiview_warp_result/";
    string maskDir = data_path + "mask/";
    string overlapDir = data_path + "overlap/";
    string seamDir = data_path + "seam/";

    Inits::loadAll(warpDir, seamDir, overlapDir, maskDir);
    height = warped_imgs[0].rows;
    width = warped_imgs[0].cols;

    Inits::Initialize_var();

    TickMeter tm;
    tm.start();

    vector<int> exe_seq = Utils::MST();

    //get final result
    Mat result_img = Utils::build_final_result();

    tm.stop();
    cout << "\nExecution time = " << tm.getTimeSec() << " s." << endl;
 
    //save result
    for (int i = 0; i < warped_imgs.size(); i++)
        imwrite(data_path + "AMWF_results/" + to_string(i) + "__warp.png", warped_imgs[i]);

    Mat hole = Utils::hole_detection(result_img);
    Utils::hole_filling(result_img, hole);
    imwrite(data_path + "stitched_results/AMWF.png", result_img);

    cout << data_path << " --> Color correction finished.\n";

    return 0;
}