#pragma once

#include "Vector2D.cuh"

struct classComparePoint {
	bool operator() (const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const{
		if (lhs.first != rhs.first) return lhs.first > rhs.first;
		return lhs.second > rhs.second;
	}
};

bool isOnLine(int pi, int pj, int pr, std::vector<std::pair<int, int>>& hashPoints) {
	//return true if pr is on the line pi -> pj
	Vec2 tempVec = pj - pi;
	if ((tempVec.x * pr.y - tempVec.y * pr.x) - 0.0 > EPSILON) return false; return true;
}

int calculate2DlineSideTest(int pi, int pj, int pr, std::vector<std::pair<int, int>>& hashPoints) {
	//return 1 if pr is on left side of vector pi -> pj
	//return -1 if on right
	//return 0 if pr is on line pi -> pj
	if ((pi == 0 && pj == -1) || (pi == -2 && pj == 0) || (pi == -1 && pj == -2)) return -1;
	if ((pj == 0 && pi == -1) || (pj == -2 && pi == 0) || (pj == -1 && pi == -2)) return 1;


	if (pj == -1 || pi == -2) return false; if (pj == -2 || pi == -1 ) return true;
	Vec2 tempVec = pj - pi;
	if (tempVec.x * pr.y - tempVec.y * pr.x < 0.0) return false; return true;
}





void legalizeEdge(Vec2 pr, Vec2 pi, Vec2 pj, TreeNode* root) {

}

void performDS(std::vector<std::pair<int, int>>& hashPoints, cv::Mat& image, cv::Mat& outImage) {
	//find P0
	//Point with the highest y and right most x
	std::multimap < std::pair<int,int>, size_t , classComparePoint > ySort;
	for (size_t i = 0; i < hashPoints.size(); ++i)
		ySort.insert(std::pair<std::pair<int,int>, size_t>(std::pair<int,int>(hashPoints[i].first, hashPoints[i].second), i));
	//for (auto& v : ySort) std::cout << v.first.first << " " << v.first.second << std::endl;
	if (ySort.size() == 0) throw "Empty ySort";
	auto it = cbegin(ySort);
	//p0 coords and index
	Vec2 p0(it->first.second, it->first.first);
	int p0Indx = it->second;
	std::cout << p0.x << std::endl;
	std::cout << p0Indx << std::endl;

	TreeNode* root = new TreeNode(p0Indx, -1, -2);

	for (int i = 0; i < hashPoints.size(); ++i) {
		if (i != p0Indx){
			//search for the triangle in the tree that contains your point and is a leaf
			//two triangles may contain your point if the point lies on a line





		}
	}


}
