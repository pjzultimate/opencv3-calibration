#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    ofstream fout("/home/pjz/vins/calib1/caliberation_result.txt");  /* 保存标定结果的文件 */
    Mat image;
    vector<Mat> images;
    int image_count=0;//图像数量
    Size image_size;//图像的尺寸
    Size board_size=Size(8,6);//标定板上每行、列的角点数
    vector<Point2f> image_points_buf;
    vector<vector<Point2f>> image_points_seq;
    int count=-1;
    for(int i=1;i<14;i++)
    {
        image_count++;
        string file = format("/home/pjz/vins/calib1/%d.tif",i);
        image=imread(file);
        if(image.empty())
        {
            cerr<<"cuowu"<<endl;
        }
        if(image_count==1)
        {
            image_size.width=image.cols;
        image_size.height=image.rows;
        cout<<image.cols<<":"<<image.rows<<endl;
        }
        if(0==findChessboardCorners(image,board_size,image_points_buf))
        {

            cout<<"can not find corners"<<endl;
            exit(1);
        }
        else
        {
            Mat view_gray;
            cvtColor(image,view_gray,CV_RGB2GRAY);
            cornerSubPix(view_gray,image_points_buf,Size(7,7),Size(-1,-1),TermCriteria(
                             CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,
                             40,
                             0.001 ));//7 7zuihao
            //find4QuadCornerSubpix(view_gray,image_points_buf,Size(5,5));// diyushangbiande 0.5pixel//5 5 zuihao
            image_points_seq.push_back(image_points_buf);
            drawChessboardCorners(view_gray,board_size,image_points_buf,true);
            imshow("camera calibratoin",view_gray);
            waitKey(500);

        }

    }
    int total=image_points_seq.size();
    cout<<total<<endl;
    int cornernum=board_size.width*board_size.height;
    for(int ii=0;ii<total;ii++)
    {
       if(0==ii%cornernum)
       {
           int i=-1;
           i=ii/cornernum;
           int j=i+1;
           cout<<"di"<<j<<"tu"<<endl;
       }
       cout<<"zuobiao"<<image_points_seq[ii][0].x<<";"<<image_points_seq[ii][0].y<<endl;
    }
    cout<<"finish corner"<<endl;
    Size square_size = Size(40,40);
    vector<vector<Point3f>>object_points;
    Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0));
    vector<int> point_counts;
    Mat distcoeffs= Mat(1,5,CV_32FC1,Scalar::all(0));
    vector<Mat>tvecsMat;
    vector<Mat>rvecsMat;
    int i,j,t;
    for (t=0;t<image_count;t++)
       {
           vector<Point3f> tempPointSet;
           for (i=0;i<board_size.height;i++)
           {
               for (j=0;j<board_size.width;j++)
               {
                   Point3f realPoint;
                   /* 假设标定板放在世界坐标系中z=0的平面上 */
                   realPoint.x = i*square_size.width;
                   realPoint.y = j*square_size.height;
                   realPoint.z = 0;
                   tempPointSet.push_back(realPoint);
               }
           }
           object_points.push_back(tempPointSet);
       }
    for (i=0;i<image_count;i++)
        {
            point_counts.push_back(board_size.width*board_size.height);
        }
        /* 开始标定 */
        calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distcoeffs,rvecsMat,tvecsMat,0);
        cout<<"标定完成！\n";
            //对标定结果进行评价
            cout<<"开始评价标定结果………………\n";
            double total_err = 0.0; /* 所有图像的平均误差的总和 */
            double err = 0.0; /* 每幅图像的平均误差 */
            vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
            cout<<"\t每幅图像的标定误差：\n";
            fout<<"每幅图像的标定误差：\n";
            for (i=0;i<image_count;i++)
            {
                vector<Point3f> tempPointSet=object_points[i];
                /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
                projectPoints(tempPointSet,rvecsMat[i],tvecsMat[i],cameraMatrix,distcoeffs,image_points2);
                /* 计算新的投影点和旧的投影点之间的误差*/
                vector<Point2f> tempImagePoint = image_points_seq[i];
                Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
                Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
                for (int j = 0 ; j < tempImagePoint.size(); j++)
                {
                    image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);
                    tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
                }
                err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
                total_err += err/=  point_counts[i];
                std::cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
                fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
            }
            std::cout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl;
            fout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;
            std::cout<<"评价完成！"<<endl;
            //保存定标结果
            std::cout<<"开始保存定标结果………………"<<endl;
            Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
            fout<<"相机内参数矩阵："<<endl;
            fout<<cameraMatrix<<endl<<endl;
            fout<<"畸变系数：\n";
            fout<<distcoeffs<<endl<<endl<<endl;
            for (int i=0; i<image_count; i++)
            {
                fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;
                fout<<tvecsMat[i]<<endl;
                /* 将旋转向量转换为相对应的旋转矩阵 */
                Rodrigues(tvecsMat[i],rotation_matrix);
                fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;
                fout<<rotation_matrix<<endl;
                fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;
                fout<<rvecsMat[i]<<endl<<endl;
            }
            std::cout<<"完成保存"<<endl;
            fout<<endl;
        //cout<<images.size()<<endl;
}
