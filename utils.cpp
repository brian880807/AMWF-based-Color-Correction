#include "utils.h"

vector<string> Inits::getImgFilenameList(string& addr)
{
    vector<string> imgs;
    for(auto& entry : filesystem::directory_iterator(addr)){
        if(entry.path().u8string().find(".png") != string::npos)
            imgs.push_back(entry.path().u8string());
    }
    return imgs;
}

vector<Mat> Inits::LoadImage(string& addr, int flag)
{
    vector<string> img_names = Inits::getImgFilenameList(addr);
    vector<Mat> imgs;
    imgs.reserve(img_names.size());
    for(const string& file_name : img_names)
        imgs.push_back(imread(file_name, flag));

    return imgs;
}

vector<vector<Mat>> Inits::LoadMask(string& addr, int N)
{
    vector<vector<int>> seq(N);
    vector<vector<Mat>> compilation(N);
    vector<string> img_names = Inits::getImgFilenameList(addr);

    for(const string& file_name : img_names){
        int num = stoi(file_name.substr(addr.size(), 2));
        seq[num].push_back(stoi(file_name.substr(addr.size() + 4, 2)));
        compilation[num].push_back(imread(file_name, IMREAD_GRAYSCALE));
    }

    cout << img_names.size();
    ref_seq = seq;
    return compilation;
}

void Inits::loadAll(string& warpDir, string& seamDir, string& overlapDir, string& maskDir)
{
    cout << "load warp images-------------------->\n\n";
    warped_imgs = Inits::LoadImage(warpDir, IMREAD_COLOR);
    int N = (int)warped_imgs.size();
    cout << N << " images loaded.\n\n";

    cout << "load seam images-------------------->\n\n";
    seams = Inits::LoadMask(seamDir, N);
    cout << " seam images loaded.\n\n";

    cout << "load overlap images----------------->\n\n";
    overlaps = Inits::LoadMask(overlapDir, N);
    cout << " overlap images loaded.\n\n";

    cout << "load mask images-------------------->\n\n";
    masks = Inits::LoadMask(maskDir, N);
    cout << " mask images loaded.\n\n";
}

vector<Point> Utils::build_seam_pixel_lst(Mat& seam_mask)
{
    vector<Point> locations;
    cv::findNonZero(seam_mask, locations);
    return locations;
}

Mat Utils::refine_seam_pixel_lst_based_abs_RGB_diff_OTSU(vector<Point>& seam_pixel_lst, Mat& warped_ref_img, Mat& warped_tar_img)
{
    int count = (int)seam_pixel_lst.size();
    Mat points1(count, 1, CV_8U);
    Mat points2(count, 1, CV_8U);
    Mat points3(count, 1, CV_8U);
    Mat label1, label2, label3;
    vector<vector<float>> out1(2), out2(2), out3(2);

    for (int i = 0; i < count; i++)
    {
        Point p = seam_pixel_lst[i];
        int distance1 = abs((int)warped_ref_img.at<Vec3b>(p)[0] - (int)warped_tar_img.at<Vec3b>(p)[0]);
        int distance2 = abs((int)warped_ref_img.at<Vec3b>(p)[1] - (int)warped_tar_img.at<Vec3b>(p)[1]);
        int distance3 = abs((int)warped_ref_img.at<Vec3b>(p)[2] - (int)warped_tar_img.at<Vec3b>(p)[2]);

        points1.at<uchar>(i, 0) = (uchar)distance1;
        points2.at<uchar>(i, 0) = (uchar)distance2;
        points3.at<uchar>(i, 0) = (uchar)distance3;
    }

    int T_1 = (int)threshold(points1, label1, 0, 255, THRESH_BINARY | THRESH_OTSU);
    int T_2 = (int)threshold(points2, label2, 0, 255, THRESH_BINARY | THRESH_OTSU);
    int T_3 = (int)threshold(points3, label3, 0, 255, THRESH_BINARY | THRESH_OTSU);

    //cout << "OTSU----->\n";

    int sum1 = (int)(sum(label1)[0] / 255.0);
    int sum2 = (int)(sum(label2)[0] / 255.0);
    int sum3 = (int)(sum(label3)[0] / 255.0);
    int target1 = sum1 < count / 2 ? 0 : 255;
    int target2 = sum2 < count / 2 ? 0 : 255;
    int target3 = sum3 < count / 2 ? 0 : 255;
    int anomaly_count = 0;
    vector<Point> discarded_seam_pixel_lst;

    for (int i = 0; i < count; i++)
    {
        if (label1.at<uchar>(i) == target1 && label2.at<uchar>(i) == target2 && label3.at<uchar>(i) == target3)
        {
            //refined_seam_pixel_lst.push_back(seam_pixel_lst[i]);
        }
        else
        {
            anomaly_count++;
            discarded_seam_pixel_lst.push_back(seam_pixel_lst[i]);
        }

        if (points1.at<uchar>(i, 0) < T_1)
            out1[0].push_back(points1.at<uchar>(i, 0));
        else
            out1[1].push_back(points1.at<uchar>(i, 0));

        if (points2.at<uchar>(i, 0) < T_2)
            out2[0].push_back(points2.at<uchar>(i, 0));
        else
            out2[1].push_back(points2.at<uchar>(i, 0));

        if (points3.at<uchar>(i, 0) < T_3)
            out3[0].push_back(points3.at<uchar>(i, 0));
        else
            out3[1].push_back(points3.at<uchar>(i, 0));
    }

    int ordinary_count = count - anomaly_count;
    double oma = (double)ordinary_count * anomaly_count;
    double out1_dif = sum(out1[0])[0] / (double)out1[0].size() - sum(out1[1])[0] / (double)out1[1].size();
    double out2_dif = sum(out2[0])[0] / (double)out2[0].size() - sum(out2[1])[0] / (double)out2[1].size();
    double out3_dif = sum(out3[0])[0] / (double)out3[0].size() - sum(out3[1])[0] / (double)out3[1].size();
    double cost1 = oma * out1_dif * out1_dif / double(count * count);
    double cost2 = oma * out2_dif * out2_dif / double(count * count);
    double cost3 = oma * out3_dif * out3_dif / double(count * count);

    if (out1[0].empty()) cost1 = 0.1;
    if (out2[0].empty()) cost2 = 0.1;
    if (out3[0].empty()) cost3 = 0.1;

    Mat anomaly_mask = Mat::zeros(height, width, CV_8U);
    if (cost1 >= cost_threshold || cost2 >= cost_threshold || cost3 >= cost_threshold)
    {
        for (Point p : discarded_seam_pixel_lst){
            anomaly_mask.at<uchar>(p) = 255;
        }
    }
    return anomaly_mask;
}

vector<Point> Utils::sort_seam_pixel_lst(vector<Point>& seam_pixel_lst, Mat& seam_mask)
{
    vector<Point> endpoints_lst;
    vector<int> x_off = {-1, 1, 0, 0};
    vector<int> y_off = {0, 0, -1, 1};
    for (Point p : seam_pixel_lst){
        int count = 0;
        for (int i = 0; i < 4; i++){
            Point np(p.x + x_off[i], p.y + y_off[i]);
            if (seam_mask.at<uchar>(np) == 255)
                count++;
        }
        if (count == 1){
            endpoints_lst.push_back(p);
        }
    }

    vector<Point> max_len_seam;
    for(Point end_point : endpoints_lst) {
        Mat seam_discover_map;
        seam_mask.copyTo(seam_discover_map);
        seam_discover_map.at<uchar>(end_point) = 0;
        stack<Point> discover_stack;
        discover_stack.push(end_point);
        vector<Point> sorted_seam_pixel_lst = {end_point};

        while (!discover_stack.empty()) {
            Point p = discover_stack.top();
            discover_stack.pop();
            for (int i = 0; i < 4; i++) {
                Point np(p.x + x_off[i], p.y + y_off[i]);
                if (seam_discover_map.at<uchar>(np) == 255) {
                    seam_discover_map.at<uchar>(np) = 0;
                    sorted_seam_pixel_lst.push_back(np);
                    discover_stack.push(np);
                }
            }
        }

        if(max_len_seam.size() < sorted_seam_pixel_lst.size())
            max_len_seam = sorted_seam_pixel_lst;
    }

    return max_len_seam;
}

void Utils::build_range_map_with_side_addition(Mat& tar_img, vector<Point>& sorted_seam_pixel_lst)
{
    range_count_map = Mat::zeros(height, width, CV_32FC1);
    for (int idx = 0; idx < sorted_seam_pixel_lst.size(); idx++)
    {
        range_map.at<Vec2w>(sorted_seam_pixel_lst[idx])[0] = idx;
        range_map.at<Vec2w>(sorted_seam_pixel_lst[idx])[1] = idx;
    }

    int max_cite_range = (int)sorted_seam_pixel_lst.size() - 1;
    int time_stamp = 1;
    static uint8_t gray = 255;
    static int x_offset[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    static int y_offset[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    bool flag_early_termination = false;
    vector<Point> next_wavefront;
    next_wavefront.reserve(10000);
    vector<Point> current_wavefront = sorted_seam_pixel_lst;

    Mat discovered_map(height, width, CV_8U);
    //Mat discovered_time_stamp_map(height, width, CV_16U);
    for (Point point : sorted_seam_pixel_lst)
        discovered_map.at<uchar>(point) = 255;
    for (Point point : sorted_seam_pixel_lst)
        discovered_time_stamp_map.at<ushort>(point) = time_stamp;

    // march through whole target image.
    while (true)
    {
        // find next wavefront
        time_stamp++;
        next_wavefront.clear();
        for (Point point : current_wavefront)
        {
            for (int i = 0; i < 8; i++)
            {
                int y_n = point.y + y_offset[i];
                int x_n = point.x + x_offset[i];
                if (x_n == -1 || y_n == -1 || x_n == width || y_n == height)
                    continue;

                if (discovered_map.at<uchar>(y_n, x_n) == 0 && tar_img.at<Vec3b>(y_n, x_n) != Vec3b::zeros())
                {
                    next_wavefront.emplace_back(x_n, y_n);
                    discovered_map.at<uchar>(y_n, x_n) = 255;
                    discovered_time_stamp_map.at<ushort>(y_n, x_n) = time_stamp;
                }
            }
        }

        // break from the while loop if there is no next wavefront.
        if (next_wavefront.empty())
            break;
        int reference_all_range_cnt = 0;
        // propagate the citation range from previous wavefront to current wavefront.
        for (Point point : next_wavefront)
        {
            int min_range = std::numeric_limits<int>::max();
            int max_range = std::numeric_limits<int>::min();

            if (!flag_early_termination)
            {
                for (int i = 0; i < 8; i++)
                {
                    int y_n = point.y + y_offset[i];
                    int x_n = point.x + x_offset[i];
                    if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height)
                        continue;
                    if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp - 1)
                    {
                        if (min_range > range_map.at<Vec2w>(y_n, x_n)[0])
                            min_range = range_map.at<Vec2w>(y_n, x_n)[0];
                        if (max_range < range_map.at<Vec2w>(y_n, x_n)[1])
                            max_range = range_map.at<Vec2w>(y_n, x_n)[1];
                    }
                }

                min_range = max(min_range - propagation_coefficient, 0);
                max_range = min(max_range + propagation_coefficient, max_cite_range);
                range_map.at<Vec2w>(point)[0] = min_range;
                range_map.at<Vec2w>(point)[1] = max_range;


                if (range_map.at<Vec2w>(point)[0] == 0 && range_map.at<Vec2w>(point)[1] == max_cite_range)
                {
                    // Fecker_mask.at<uchar>(point) = 255;
                    reference_all_range_cnt++;
                }

                range_count_map.at<int>(point) = max_range - min_range + 1;
            }
            else
            {
                range_map.at<Vec2w>(point)[0] = 0;
                range_map.at<Vec2w>(point)[1] = max_cite_range;
                range_count_map.at<int>(point) = max_range - min_range + 1;
            }
        }
        if (next_wavefront.size() == reference_all_range_cnt)
            flag_early_termination = true;

        vector<int> pending_update_min_ranges, pending_update_max_ranges;

        for (Point point : next_wavefront)
        {
            int min_range = std::numeric_limits<int>::max();
            int max_range = std::numeric_limits<int>::min();

            for (int i = 0; i < 8; i++)
            {
                int y_n = point.y + y_offset[i];
                int x_n = point.x + x_offset[i];
                if (x_n == -1 || y_n == -1 || x_n >= width || y_n >= height)
                    continue;
                if (discovered_time_stamp_map.at<ushort>(y_n, x_n) == time_stamp)
                {
                    if (min_range > range_map.at<Vec2w>(y_n, x_n)[0])
                        min_range = range_map.at<Vec2w>(y_n, x_n)[0];
                    if (max_range < range_map.at<Vec2w>(y_n, x_n)[1])
                        max_range = range_map.at<Vec2w>(y_n, x_n)[1];
                }
            }
            pending_update_min_ranges.push_back(min_range);
            pending_update_max_ranges.push_back(max_range);
        }

        for (int idx = 0; idx < next_wavefront.size(); idx++)
        {
            range_map.at<Vec2w>(next_wavefront[idx])[0] = pending_update_min_ranges[idx];
            range_map.at<Vec2w>(next_wavefront[idx])[1] = pending_update_max_ranges[idx];
        }

        gray -= 10;

        current_wavefront = next_wavefront;
    }

    //discovered_time_stamp_map.copyTo(dts_map);
    //return range_count_map;
}

vector<Point> Utils::build_target_pixel_lst(Mat& tar_img)
{
    vector<Point> locations;
    Mat gray;
    cvtColor(tar_img, gray, COLOR_RGB2GRAY);
    cv::findNonZero(gray, locations);

    return locations;
}

void Utils::init_color_comp_map(vector<Point>& seam_pixel_lst, Mat& warped_ref_img, Mat& warped_tar_img)
{
    for(Point& p : seam_pixel_lst){
        color_comp_map.at<Vec3d>(p)[0] = double(warped_ref_img.at<Vec3b>(p)[0]) - double(warped_tar_img.at<Vec3b>(p)[0]);
        color_comp_map.at<Vec3d>(p)[1] = double(warped_ref_img.at<Vec3b>(p)[1]) - double(warped_tar_img.at<Vec3b>(p)[1]);
        color_comp_map.at<Vec3d>(p)[2] = double(warped_ref_img.at<Vec3b>(p)[2]) - double(warped_tar_img.at<Vec3b>(p)[2]);
    }
}

void Utils::update_color_comp_map_range_anomaly(vector<Point>& pxl_lst, Mat& warped_tar_img, vector<Point>& sorted_seam_pixel_lst, Mat& anomaly_mask)
{
    #pragma omp parallel for
    for(int i = 0; i < pxl_lst.size(); i++){
        vector<double> color_comp = get_color_comp_value_range_anomaly(pxl_lst[i], warped_tar_img, sorted_seam_pixel_lst,anomaly_mask);
        color_comp_map.at<Vec3d>(pxl_lst[i])[0] = color_comp[0];
        color_comp_map.at<Vec3d>(pxl_lst[i])[1] = color_comp[1];
        color_comp_map.at<Vec3d>(pxl_lst[i])[2] = color_comp[2];
    }
}

vector<double> Utils::get_color_comp_value_range_anomaly(Point& p, Mat& warped_tar_img, vector<Point>& sorted_seam_pixel_lst, Mat& anomaly_mask)
{
    vector<double> color_comp = { 0.0, 0.0, 0.0 };
    int low_bound = range_map.at<Vec2w>(p)[0];
    int high_bound = range_map.at<Vec2w>(p)[1];
    int range_num = high_bound - low_bound + 1;
    int total_anomaly_cnt = 0;

    for (int i = 0; i < range_num; i++){
        int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
        int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
        if (anomaly_mask.at<uchar>(y_seam, x_seam) == 255) total_anomaly_cnt++;
    }

    double anomaly_ratio = (double)total_anomaly_cnt / (double)range_num;

    double weight_sum = 0.0;
    for (int i = 0; i < range_num; i++){
        int x_seam = sorted_seam_pixel_lst[i + low_bound].x;
        int y_seam = sorted_seam_pixel_lst[i + low_bound].y;
        double dist_diff = (p.x - x_seam) * (p.x - x_seam) + (p.y - y_seam) * (p.y - y_seam);
        double color_diff = (pow((double(warped_tar_img.at<Vec3b>(p)[0]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[0])), 2)
            + pow((double(warped_tar_img.at<Vec3b>(p)[1]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[1])), 2)
            + pow((double(warped_tar_img.at<Vec3b>(p)[2]) - double(warped_tar_img.at<Vec3b>(y_seam, x_seam)[2])), 2)) / 65025;
        double new_color_sigma = max(sigma_color * anomaly_ratio, min_sigma_color);
        double a = color_diff / new_color_sigma / new_color_sigma * -1.0;
        double b = dist_diff / sigma_dist / sigma_dist / width / width * -1.0;
        double weight = exp(a) * exp(b);
        color_comp[0] += color_comp_map.at<Vec3d>(y_seam, x_seam)[0] * weight;
        color_comp[1] += color_comp_map.at<Vec3d>(y_seam, x_seam)[1] * weight;
        color_comp[2] += color_comp_map.at<Vec3d>(y_seam, x_seam)[2] * weight;
        weight_sum += weight;
    }
    if(weight_sum == 0.0){
        color_comp[0] = 0.0;
        color_comp[1] = 0.0;
        color_comp[2] = 0.0;
    }
    else{
        color_comp[0] = color_comp[0] / weight_sum;
        color_comp[1] = color_comp[1] / weight_sum;
        color_comp[2] = color_comp[2] / weight_sum;
    }

    return color_comp;
}

//Fecker func
ImgPack Utils::CalDF(Mat& src, int channel, Mat& overlap)
{
    ImgPack src_DF;
    src_DF.PDF.assign(256, 0);
    src_DF.CDF.assign(256, 0);

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            if (overlap.at<bool>(i, j))
                src_DF.PDF[(int)src.at<Vec3b>(i, j)[channel]]++;

    src_DF.CDF[0] = src_DF.PDF[0];

    for (int i = 1; i < src_DF.PDF.size(); i++)
        src_DF.CDF[i] = src_DF.PDF[i] + src_DF.CDF[i - 1];

    return src_DF;
}

Mat Utils::HM_Fecker(Mat& ref, Mat& tar, Mat& overlap)
{
    vector<vector<int>> mapping_Func(3, vector<int>(256, 0));
    for (int channel = 0; channel < 3; channel++) {
        ImgPack ref_DF, tar_DF;
        ref_DF = CalDF(ref, channel, overlap);
        tar_DF = CalDF(tar, channel, overlap);

        bool flag = false;
        int temp_x = -100, temp_y = -100;
        for (int i = 0; i < 256; i++){
            for (int j = 0; j < 256; j++){
                if (ref_DF.CDF[j] > tar_DF.CDF[i]){
                    if (!flag && (j - 1) >= 0)
                    {
                        flag = true;
                        temp_x = i;
                        temp_y = j;
                    }

                    mapping_Func[channel][i] = (int)saturate_cast<uchar>(j);
                    break;
                }

                if(i) mapping_Func[channel][i] = mapping_Func[channel][i - 1];
            }
        }

        for (int i = temp_x; i >= 0; i--)
            mapping_Func[channel][i] = temp_y;

        float sum1 = 0, sum2 = 0;
        for (int i = 0; i <= mapping_Func[channel][0]; i++)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * (float)i;
        }
        mapping_Func[channel][0] = (int)(sum2 / sum1);

        sum1 = 0, sum2 = 0;
        for (int i = 255; i >= mapping_Func[channel][255]; i--)
        {
            sum1 += ref_DF.PDF[i];
            sum2 += ref_DF.PDF[i] * (float)i;
        }
        mapping_Func[channel][255] = (int)(sum2 / sum1);
    }

    Mat output_comp = Mat(ref.size(), ref.type());
    for (int channel = 0; channel < 3; channel++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                bool sw = tar.at<Vec3b>(i, j)[0] != 0 || tar.at<Vec3b>(i, j)[1] != 0 || tar.at<Vec3b>(i, j)[2] != 0;
                if(sw)
                    output_comp.at<Vec3b>(i, j)[channel] = mapping_Func[channel][tar.at<Vec3b>(i, j)[channel]];
                else
                    output_comp.at<Vec3b>(i, j)[channel] = tar.at<Vec3b>(i, j)[channel];
            }
        }
    }

    tar.convertTo(tar, CV_64FC3);
    output_comp.convertTo(output_comp, CV_64FC3);
    output_comp -= tar;
    tar.convertTo(tar, CV_8UC3);

    return output_comp;
}

void Inits::Initialize_var()
{
    // initialize map
    discovered_time_stamp_map = Mat::zeros(height, width, CV_16U);
    range_map = Mat::zeros(height, width, CV_16UC2);
    color_comp_map = Mat::zeros(height, width, CV_64FC3);
}

void Utils::getFeckerWeightingCorrectedImg(Mat& warped_tar_img, int num, vector<Mat>& cites_range, vector<Mat>& wave_num,
                                           vector<Mat>& comps, vector<Mat>& Fecker_comps, vector<double>& seam_lengths)
{
    if(cites_range.empty()) return;

    Mat total_wave = Mat::zeros(wave_num[0].size(), wave_num[0].type());
    for (const Mat& wave : wave_num)
        total_wave += wave;

    warped_tar_img.convertTo(warped_tar_img, CV_64FC3);

    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            Point p(x, y);
            if (!masks[num][0].at<bool>(p) && !overlaps[num][0].at<bool>(p))
                continue;

            double n_2 = 0.0, n_2_normalized = 0.0;
            for (int i = 0; i < Fecker_comps.size(); i++)
                n_2 += wave_num[i].at<ushort>(p);

            for (int i = 0; i < Fecker_comps.size(); i++)
                n_2_normalized += n_2 / wave_num[i].at<ushort>(p);

            if (n_2 == 0) continue;
            for (int i = 0; i < Fecker_comps.size(); i++){
                for (int c = 0; c < 3; c++) {
                    double rate = 1.0 * cites_range[i].at<int>(p) / seam_lengths[i];
                    rate = rate > 1.0 ? 1.0 : rate < 0.0 ? 0.0 : rate;
                    rate = Fecker_comps[i].at<Vec3d>(p)[c] * rate + comps[i].at<Vec3d>(p)[c] * (1.0 - rate);
                    warped_tar_img.at<Vec3d>(p)[c] += rate * (n_2 / (wave_num[i].at<ushort>(p)) / n_2_normalized);
                }
            }
        }
    }
    warped_tar_img.convertTo(warped_tar_img, CV_8UC3);
}

Mat Utils::build_final_result()
{
    Mat result, mask_tmp;
    for(int i = 0; i < warped_imgs.size(); i++){
        masks[i][0].copyTo(mask_tmp);
        for(int j = 1; j < masks[i].size(); j++){
            bitwise_and(mask_tmp, masks[i][j], mask_tmp);
        }
        warped_imgs[i].copyTo(result, mask_tmp);
    }

    return result;
}

Mat Utils::hole_detection(Mat& result_img)
{
    Mat hole = Mat(masks[0][0].size(), masks[0][0].type());
    for (int x = 0; x < height; x++)
    {
        for (int y = 0; y < width; y++)
        {
            if (result_img.at<Vec3b>(x, y)[0]==0 && result_img.at<Vec3b>(x, y)[1]==0 && result_img.at<Vec3b>(x, y)[2]==0)
                hole.at<uchar>(x, y) = 255;
        }
    }

    return hole;
}

void Utils::hole_filling(Mat& result_img, Mat& hole)
{
    Mat filling = Mat(warped_imgs[0].size(), warped_imgs[0].type());
    vector<Point> location;
    findNonZero(hole, location);
    for (Point& p : location)
    {
        vector<double> color(3, 0.0);
        int count = 0;
        for(int i = 0; i < warped_imgs.size(); i++){
            if (!masks[i][0].at<bool>(p) && !overlaps[i][0].at<bool>(p))
                continue;

            color[0] += warped_imgs[i].at<Vec3b>(p)[0];
            color[1] += warped_imgs[i].at<Vec3b>(p)[1];
            color[2] += warped_imgs[i].at<Vec3b>(p)[2];
            count++;
        }
        for (int i = 0; i < color.size(); i++)
        {
            color[i] /= count;
            filling.at<Vec3b>(p)[i] = saturate_cast<uchar>(color[i]);
        }   
    }

    filling.copyTo(result_img, hole);
}

int Utils::getNextImageID(const vector<bool>& is_corrected)
{
    //check if all images are corrected
    bool is_all = true;
    for(bool cor : is_corrected){
        if(!cor){
            is_all = false;
            break;
        }
    }
    if(is_all) return -1;

    //convert RGB warped images into grayscale images
    vector<Mat> warped_gray(warped_imgs.size());
    for(int i = 0; i < warped_imgs.size(); i++)
        cvtColor(warped_imgs[i], warped_gray[i], COLOR_BGR2GRAY);

    //get {MSE, img_id1, img_id2} array
    vector<vector<double>> overlap_mse;
    vector<double> edge_sums(warped_imgs.size(), 0.0f);
    for(int i = 0; i < overlaps.size(); i++){
        for(int j = 0; j < overlaps[i].size(); j++){
            int ref = ref_seq[i][j];
            if(i < ref) {
                vector<Point> overlap_indices;
                findNonZero(overlaps[i][j], overlap_indices);

                //calculate the modified color distance (MCD) of overlapping area
                double cd = 0.0;
                vector<vector<int>> tgt_hist(3, vector<int>(256, 0));
                vector<vector<int>> ref_hist(3, vector<int>(256, 0));
                for (Point &p: overlap_indices) {
                    for (int ch = 0; ch < 3; ch++) {
                        tgt_hist[ch][warped_imgs[i].at<Vec3b>(p)[ch]]++;
                        ref_hist[ch][warped_imgs[ref].at<Vec3b>(p)[ch]]++;
                    }
                }

                for (int ch = 0; ch < 3; ch++) {
                    for (int val = 0; val < 256; val++) {
                        cd += abs(tgt_hist[ch][val] - ref_hist[ch][val]);
                    }
                }
                cd /= (double) overlap_indices.size();
                overlap_mse.push_back({cd, (double) i, (double) ref});
                edge_sums[i] += cd;
                edge_sums[ref] += cd;
            }
        }
    }

    //get the next index of image to correct (start from the image pair with the lowest MSE/CD)
    int next = -1;
    std::sort(overlap_mse.begin(), overlap_mse.end());
    for(vector<double>& mses : overlap_mse){
        int id1 = (int)mses[1], id2 = (int)mses[2];
        //if both image are corrected, check the next image pair
        if(is_corrected[id1] && is_corrected[id2])
            continue;
        else if(is_corrected[id1]){
            next = id2;
            break;
        }
        else if(is_corrected[id2]){
            next = id1;
            break;
        }
        else{
            //if both image are not corrected, correct the image with smaller edge weight sum
            next = edge_sums[id1] < edge_sums[id2] ? id1 : id2;
            break;
        }
    }

    return next;
}

vector<int> Utils::MST()
{
    //get the first image to correct
    vector<bool> is_corrected(warped_imgs.size(), false);
    int i = Utils::getNextImageID(is_corrected);

    vector<int> final_sequence;
    while(i != -1)
    {
        cout << "\n******* Correcting image " << i << " *******\n";
        final_sequence.push_back(i);

        Utils::single_img_correction(i, is_corrected);
        is_corrected[i] = true;

        //get the next image to correct
        i = Utils::getNextImageID(is_corrected);
    }

    cout << "\nCorrection sequence: ";
    for(int& seq : final_sequence) cout << seq << " ";
    cout << endl;

    return final_sequence;
}

void Utils::single_img_correction(int i, const vector<bool>& is_corrected)
{
    vector<Mat> cites_range, wave_num, comps, Fecker_comps;
    vector<double> seam_lengths;
    for (int j = 0; j < ref_seq[i].size(); j++)
    {
        Inits::Initialize_var();
        cout << "\nCalculating HEJBI terms for Target image: " << i << " and source image: " << ref_seq[i][j] << endl;

        // fusion color transfer
        // AJBI-based
        vector<Point> seam_pixel_lst = Utils::build_seam_pixel_lst(seams[i][j]);
        if(seam_pixel_lst.size() < 5){
            continue;
        }
        Mat anomaly_mask = Utils::refine_seam_pixel_lst_based_abs_RGB_diff_OTSU(seam_pixel_lst, warped_imgs[ref_seq[i][j]], warped_imgs[i]);
        vector<Point> sorted_seam_pixel_lst = Utils::sort_seam_pixel_lst(seam_pixel_lst, seams[i][j]);
        seam_lengths.push_back((double)sorted_seam_pixel_lst.size());
        Utils::build_range_map_with_side_addition(warped_imgs[i], sorted_seam_pixel_lst);
        // fusion2 weight(wavefront)
        wave_num.push_back(discovered_time_stamp_map);

        // fusion1 weight(cite_range)
        cites_range.push_back(range_count_map);

        vector<Point> target_pixel_lst = Utils::build_target_pixel_lst(warped_imgs[i]);
        Utils::init_color_comp_map(seam_pixel_lst, warped_imgs[ref_seq[i][j]], warped_imgs[i]);
        Utils::update_color_comp_map_range_anomaly(target_pixel_lst, warped_imgs[i], sorted_seam_pixel_lst, anomaly_mask);
        comps.push_back(color_comp_map.clone());

        // HM-based
        Mat Fecker_color_comp_map = Utils::HM_Fecker(warped_imgs[ref_seq[i][j]], warped_imgs[i], overlaps[i][j]);
        Fecker_comps.push_back(Fecker_color_comp_map);
    }

    //correct target image
    Utils::getFeckerWeightingCorrectedImg(warped_imgs[i], i, cites_range, wave_num, comps, Fecker_comps, seam_lengths);
}