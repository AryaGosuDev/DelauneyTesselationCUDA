#pragma once
#include <math.h>
#include <unordered_map>
#include <random>
#include <map>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>

#define EPSILON .000001

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

struct Triangle {
	inline Triangle() { }
	inline Triangle(Vec2& v1, Vec2& v2, Vec2& v3) {
		this->v1 = v1; this->v2 = v2; this->v3 = v3;
	}
	inline Vec2 operator + (const Vec2& v) {
		Vec2 vTemp; vTemp.x = v1.x - v2.x; vTemp.y = v1.y - v2.y; return vTemp;
	}
	Vec2 v1; Vec2 v2; Vec2 v3;
	int v1_indx; int v2_indx; int v3_indx;
	bool isInternal;
	TreeNode* parent;
	std::vector<Triangle*> adjList;
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
inline Circle createCircle(const Triangle& t) {
	Circle c;
	Vec2 baStart; baStart.x = t.v1.x; baStart.y = t.v1.y;
	Vec2 caStart; caStart.x = t.v1.x; caStart.y = t.v1.y;
	Vec2 baDir(t.v2.x - t.v1.x, t.v2.y - t.v1.y);
	Vec2 caDir(t.v3.x - t.v1.x, t.v3.y - t.v3.y);
	baStart.x += 0.5 * baDir.x; baStart.y += 0.5 * baDir.y;
	caStart.x += 0.5 * caDir.x; caStart.y += 0.5 * caDir.y;
	double t1 = 0.0, t2 = 0.0;
	Vec2 b(caStart.x - baStart.x, caStart.y - baStart.y);
	double inverseDet = baDir.x * (-1.0 * caDir.y) - baDir.y * (-1.0 * caDir.x);
	if (inverseDet == 0.0) { c.r = 0; return c; }
	else inverseDet = 1.0 / inverseDet;
	t1 = (-1.0 * caDir.y) * b.x + (caDir.x) * b.y; t1 /= inverseDet;
	c.c.x = baStart.x + t1 * baDir.x; c.c.y = baStart.y + t1 * baDir.y;
	c.r = distanceEqnt(c.c, t.v1);
	return c;
}

struct TreeNode {
	TreeNode() = default;
	TreeNode(Triangle _T) : triangle(_T) {}
	TreeNode(int _v1, int _v2, int _v3) {
		triangle.v1_indx = _v1;
		triangle.v2_indx = _v2;
		triangle.v3_indx = _v3;
		triangle.isInternal = false;
	}

	Triangle triangle;
	TreeNode* links[3];
};

Vec2 operator + (const Vec2& v1, const Vec2& v2) {
	return Vec2 (v1.x + v2.x, v1.y + v2.y);
}
Vec2 operator -(const Vec2& v1, const Vec2& v2) {
	return Vec2(v1.x - v2.x, v1.y - v2.y);
}