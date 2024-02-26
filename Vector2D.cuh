#pragma once
#include <math.h>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <random>
#include <queue>
#include <map>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <ostream>
#include <fstream>

constexpr double EPSILON = 0.00001;
constexpr int LINKS_SIZE = 3;

static int labelIter = 0;
struct TreeNode;

struct Vec2_CU {
	__device__ Vec2_CU() { x = 0; y = 0; }
	__device__ Vec2_CU(double a, double b) { x = a; y = b; }
	__device__ Vec2_CU(int _a, int _b) { x = (double)_a; y = (double)_b; }
	__device__ Vec2_CU(Vec2_CU& v) { this->x = v.x; this->y = v.y; }
	__device__ Vec2_CU& operator= (const Vec2_CU& v) { this->x = v.x; this->y = v.y; return *this; }
	__device__ void normalizeVec2() {
		double normFactor = sqrt(x * x + y * y); if (normFactor == 0.0) return; x /= normFactor; y /= normFactor;
	}
	double x;
	double y;
};

struct Vec2 {
	inline Vec2() { x = 0; y = 0; }
	inline Vec2(double a, double b) { x = a; y = b; }
	inline Vec2(Vec2& v) { this->x = v.x; this->y = v.y; }
	inline Vec2& operator= (const Vec2& v) { this->x = v.x; this->y = v.y; return *this; }
	inline void normalizeVec2() {
		double normFactor = sqrt(x * x + y * y); if (normFactor == 0.0) return; x /= normFactor; y /= normFactor;
	}
	double x;
	double y;
};

struct Triangle_CU {
	__device__ Triangle_CU() { }
	__device__ Triangle_CU(const Triangle_CU& other) { v1_indx = other.v1_indx; v2_indx = other.v2_indx; v3_indx = other.v3_indx; }
	__device__ Triangle_CU(Vec2_CU& v1, Vec2_CU& v2, Vec2_CU& v3) {
		this->v1 = v1; this->v2 = v2; this->v3 = v3;
	}
	__device__ Triangle_CU(unsigned int& a, unsigned int& b, unsigned int& c) {
		this->commutativeHash = 0;
		if (a < b && a < c) {
			this->v1_indx = a;
			this->commutativeHash += a;
			if (b < c) { this->commutativeHash += b * 1000; this->commutativeHash += c * 1000000; this->v2_indx = b;  this->v3_indx = c;}
			else { this->commutativeHash += c * 1000; this->commutativeHash += b * 1000000; this->v2_indx = c;  this->v3_indx = b; }
		}
		else if (b < a && b < c) {
			this->v1_indx = b;
			this->commutativeHash += b;
			if (a < c) { this->commutativeHash += a * 1000; this->commutativeHash += c * 1000000; this->v2_indx = a;  this->v3_indx = c;}
			else { this->commutativeHash += c * 1000; this->commutativeHash += a * 1000000; this->v2_indx = c;  this->v3_indx = a;}
		}
		else {
			this->v1_indx = c;
			this->commutativeHash += c;
			if (a < b) { this->commutativeHash += a * 1000; this->commutativeHash += b * 1000000; this->v2_indx = a;  this->v3_indx = b;}
			else { this->commutativeHash += b * 1000; this->commutativeHash += a * 1000000; this->v2_indx = b;  this->v3_indx = a;}
		}
	}
	__device__ Triangle_CU& operator= (const Triangle_CU& other) {
		v1_indx = other.v1_indx; v2_indx = other.v2_indx; v3_indx = other.v3_indx; this->commutativeHash = other.commutativeHash;
		return *this;
	}
	__device__ Vec2_CU operator + (const Vec2_CU& v) {
		Vec2_CU vTemp; vTemp.x = v1.x - v2.x; vTemp.y = v1.y - v2.y; return vTemp;
	}
	uint32_t commutativeHash;
	Vec2_CU v1; Vec2_CU v2; Vec2_CU v3;
	int v1_indx; int v2_indx; int v3_indx;
};

struct Triangle {
	inline Triangle() { }
	inline Triangle( const Triangle & other){ v1_indx = other.v1_indx; v2_indx = other.v2_indx; v3_indx = other.v3_indx; this->commutativeHash = other.commutativeHash;}
	inline Triangle( const Triangle_CU& other) {
		v1_indx = other.v1_indx; v2_indx = other.v2_indx; v3_indx = other.v3_indx; this->commutativeHash = other.commutativeHash;
	}
	inline Triangle(Vec2& v1, Vec2& v2, Vec2& v3) {
		this->v1 = v1; this->v2 = v2; this->v3 = v3;
	}
	inline Triangle(unsigned int a, unsigned int b, unsigned int c) {
		this->commutativeHash = 0;
		if (a < b && a < c) {
			this->v1_indx = a;
			this->commutativeHash += a;
			if (b < c) { this->commutativeHash += b * 1000; this->commutativeHash += c * 1000000; this->v2_indx = b;  this->v3_indx = c; }
			else { this->commutativeHash += c * 1000; this->commutativeHash += b * 1000000; this->v2_indx = c;  this->v3_indx = b; }
		}
		else if (b < a && b < c) {
			this->v1_indx = b;
			this->commutativeHash += b;
			if (a < c) { this->commutativeHash += a * 1000; this->commutativeHash += c * 1000000; this->v2_indx = a;  this->v3_indx = c; }
			else { this->commutativeHash += c * 1000; this->commutativeHash += a * 1000000; this->v2_indx = c;  this->v3_indx = a; }
		}
		else {
			this->v1_indx = c;
			this->commutativeHash += c;
			if (a < b) { this->commutativeHash += a * 1000; this->commutativeHash += b * 1000000; this->v2_indx = a;  this->v3_indx = b; }
			else { this->commutativeHash += b * 1000; this->commutativeHash += a * 1000000; this->v2_indx = b;  this->v3_indx = a; }
		}
	}
	inline Triangle& operator= (const Triangle& other) {
		v1_indx = other.v1_indx; v2_indx = other.v2_indx; v3_indx = other.v3_indx;
		return *this;
	}
	inline Vec2 operator + (const Vec2& v) {
		Vec2 vTemp; vTemp.x = v1.x - v2.x; vTemp.y = v1.y - v2.y; return vTemp;
	}
	uint32_t commutativeHash;
	Vec2 v1; Vec2 v2; Vec2 v3;
	int v1_indx; int v2_indx; int v3_indx;
};

struct Circle {
	inline Circle() {}
	inline Circle(double _r, Vec2& a) { r = _r; c = a; }
	double r;
	Vec2 c;
};

//distance eqtn
inline double distanceEqnt(const Vec2& v1, const Vec2& v2) {
	return sqrt((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y));
}

//return the circle where the vertices of the triangle lie on the edge of the circle
inline Circle createCircle(const Triangle& t, const std::vector<std::pair<int, int>>& hashPoints) {
	Circle c;
	Vec2 baStart(hashPoints[t.v1_indx].second, hashPoints[t.v1_indx].first);
	Vec2 caStart(hashPoints[t.v1_indx].second, hashPoints[t.v1_indx].first);
	Vec2 baDir(hashPoints[t.v2_indx].second - hashPoints[t.v1_indx].second, hashPoints[t.v2_indx].first - hashPoints[t.v1_indx].first);
	Vec2 caDir(hashPoints[t.v3_indx].second - hashPoints[t.v1_indx].second, hashPoints[t.v3_indx].first - hashPoints[t.v1_indx].first);
	baStart.x += 0.5 * baDir.x; baStart.y += 0.5 * baDir.y;
	caStart.x += 0.5 * caDir.x; caStart.y += 0.5 * caDir.y;
	double t1 = 0.0, t2 = 0.0;
	Vec2 b(caStart.x - baStart.x, caStart.y - baStart.y);
	Vec2 perpBADir(-1.0 * baDir.y, baDir.x); Vec2 perpCADir(-1.0 * caDir.y, caDir.x);
	double inverseDet = perpBADir.x * -perpCADir.y - (perpBADir.y * -perpCADir.x);
	if (inverseDet == 0.0) { c.r = 0; return c; }
	else inverseDet = 1.0 / inverseDet;
	t1 = inverseDet * (-perpCADir.y * b.x + perpCADir.x * b.y);
	c.c.x = baStart.x + t1 * perpBADir.x; c.c.y = baStart.y + t1 * perpBADir.y;
	Vec2 r(hashPoints[t.v1_indx].second, hashPoints[t.v1_indx].first);
	c.r = distanceEqnt(c.c, r);
	return c;
}

//TODO : delete tree recursivley using shared pointers
struct TreeNode {
	TreeNode() = default;
	TreeNode(Triangle _T) : triangle(_T) {}
	TreeNode(int _v1, int _v2, int _v3, TreeNode * _p) {
		triangle.v1_indx = _v1;
		triangle.v2_indx = _v2;
		triangle.v3_indx = _v3;
		isInternal = false;
		parent.push_back(_p);
		links[0] = links[1] = links[2] = NULL;
		label = labelIter++;
	}

	void pushParent(TreeNode* _p) { parent.push_back(_p); }

	bool isInternal;
	std::vector<TreeNode*> parent;
	int label;
	Triangle triangle;
	TreeNode* links[3];
	std::vector<TreeNode*> adjList;
	std::vector<TreeNode*> adjSiblings;
	cv::Vec3b color;
};

Vec2 operator + (const Vec2& v1, const Vec2& v2) {
	return Vec2 (v1.x + v2.x, v1.y + v2.y);
}
Vec2 operator -(const Vec2& v1, const Vec2& v2) {
	return Vec2(v1.x - v2.x, v1.y - v2.y);
}
inline double operator^(const Vec2& A, const Vec2& B){
	return A.x * B.y - A.y * B.x;
}