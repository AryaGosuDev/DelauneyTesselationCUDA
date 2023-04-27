#pragma once

#include "Vector2D.cuh"


std::ostream& operator << (std::ostream& os,  const std::pair<int,int>& p) {
	os << -p.first << "," << p.second;
	return os;
}

void extractUV(double & u, double & v, const cv::Point & pt, TreeNode * root, const std::vector<std::pair<int, int>>& hashPoints) {

	Triangle& tri = root->triangle;
	Vec2 v2v1 (hashPoints[tri.v2_indx].second - hashPoints[tri.v1_indx].second,
			-hashPoints[tri.v2_indx].first + hashPoints[tri.v1_indx].first);
	Vec2 v3v1(hashPoints[tri.v3_indx].second - hashPoints[tri.v1_indx].second,
			-hashPoints[tri.v3_indx].first + hashPoints[tri.v1_indx].first);
	Vec2 v3v2(hashPoints[tri.v3_indx].second - hashPoints[tri.v2_indx].second,
			-hashPoints[tri.v3_indx].first + hashPoints[tri.v2_indx].first);
	Vec2 v1v3(hashPoints[tri.v1_indx].second - hashPoints[tri.v3_indx].second,
			-hashPoints[tri.v1_indx].first + hashPoints[tri.v3_indx].first);
	double area = (v3v1 ^ v2v1) / 2.0;

	/*
	// edge 1
	Vec3f edge1 = v2 - v1;
	Vec3f vp1 = P - v1;
	C = edge1.crossProduct(vp1);
	u = (C.length() / 2) / area;
	if (N.dotProduct(C) < 0)  return false; // P is on the right side

	// edge 2
	Vec3f edge2 = v0 - v2;
	Vec3f vp2 = P - v2;
	C = edge2.crossProduct(vp2);
	v = (C.length() / 2) / area;
	if (N.dotProduct(C) < 0) return false; // P is on the right side;
	*/

	Vec2 prv2(pt.x - hashPoints[tri.v2_indx].second,
		pt.y + hashPoints[tri.v2_indx].first);

	u = ((prv2 ^ v3v2) / 2.0) / area;

	Vec2 prv3(pt.x - hashPoints[tri.v3_indx].second,
		pt.y + hashPoints[tri.v3_indx].first);

	v = ((prv3 ^ v1v3) / 2.0) / area;
}

void createDOTFile(TreeNode* root, const std::vector<std::pair<int, int>>& hashPoints, int indx) {

	std::unordered_multimap<TreeNode*, TreeNode* > CFGTree;
	std::unordered_map<TreeNode*, TreeNode * > CFGTreeAdj;
	std::unordered_map<TreeNode*, int > CFGLabels;
	std::unordered_set<TreeNode*> disjointVisited;
	
	std::ofstream myfile;
	std::queue<TreeNode* > q;
	myfile.open("dot.dot");
	myfile << "digraph { \n";
	myfile << "label=\"" << hashPoints[indx+1] << "\";\n";
	int cnt = 0;
	CFGLabels[NULL] = -1;
	CFGLabels[root] = cnt;
	q.push(root);
	while (!q.empty()) {
		auto top = q.front();
		
		if (CFGLabels.find(top) == end(CFGLabels)) CFGLabels[top] = ++cnt;
		if (top->isInternal == true) {
			for (int i = 0; i < LINKS_SIZE; ++i) {
				if (top->links[i] != NULL) {
					CFGTree.insert({ top, top->links[i] });
					if (disjointVisited.find(top->links[i]) == end(disjointVisited)) {
						q.push(top->links[i]);
						disjointVisited.insert(top->links[i]);
					}
				}
			}
		}
		else CFGTree.insert({ top, NULL });
		disjointVisited.insert(top);
		q.pop();
	}

	for (auto& v : CFGLabels) {
		TreeNode* root = v.first;
		if (root != NULL) {
			myfile << CFGLabels[root] << " [label=\"";
			myfile << root->label << "\n";
			if ( root->isInternal == false ) myfile << "leaf\n";
			else myfile << "internal\n";
			if (root->triangle.v1_indx == -1 || root->triangle.v1_indx == -2)
				myfile << root->triangle.v1_indx << "\n";
			else myfile << hashPoints[root->triangle.v1_indx] << "\n";
			if (root->triangle.v2_indx == -1 || root->triangle.v2_indx == -2)
				myfile << root->triangle.v2_indx << "\n";
			else myfile << hashPoints[root->triangle.v2_indx] << "\n";
			if (root->triangle.v3_indx == -1 || root->triangle.v3_indx == -2)
				myfile << root->triangle.v3_indx ;
			else myfile << hashPoints[root->triangle.v3_indx] ;
			myfile << "\"];";
		}
	}

	for (auto& v : CFGTree) {
		myfile << CFGLabels[v.first] << "->" << CFGLabels[v.second] << ";\n";
	}
	/*
	for (auto& v : CFGLabels) {
		TreeNode* root = v.first;
		if (root != NULL) {
			for (int i = 0; i < root->adjList.size(); ++i) {
				myfile << CFGLabels[root] << "->" << CFGLabels[root->adjList[i]] << " [label=a,style=dotted, arrowhead=none, color=blue];\n";
			}
			for (int i = 0; i < root->adjSiblings.size(); ++i) {
				myfile << CFGLabels[root] << "->" << CFGLabels[root->adjSiblings[i]] << " [label=s,style=dotted, arrowhead=none, color=blue];\n";
			}
		}
	}
		*/
	myfile << "}";
	myfile.close();
}

void triangleImage(TreeNode* root, cv::Mat imgO, int indx, const std::vector<std::pair<int, int>>& hashPoints) {

	cv::Mat imageOut(600, 600, CV_8UC3);

	std::unordered_set<TreeNode* > hashTree;
	std::vector<Triangle> tris;
	std::queue<TreeNode* > q;
	q.push(root);
	while (!q.empty()) {

		auto top = q.front();
		if (top->isInternal == true) {
			for (int i = 0; i < LINKS_SIZE; ++i) { if (top->links[i] != NULL) q.push(top->links[i]); }
		}
		else {
			if (hashTree.find(top) == end(hashTree)) {
				hashTree.insert(top);
				tris.push_back(top->triangle);
			}
		}
		q.pop();
	}

	for (int i = 0; i < tris.size(); ++i) {
		Triangle & t = tris[i];
		if (t.v1_indx == -1 || t.v1_indx == -2 || t.v2_indx == -1 || t.v2_indx == -2 || t.v3_indx == -1 || t.v3_indx == -2) continue;
		std::vector<cv::Point> pts;
		cv::Point pt; pt.y = -hashPoints[t.v1_indx].first; pt.x = hashPoints[t.v1_indx].second; pts.push_back(pt);
		pt.y = -hashPoints[t.v2_indx].first; pt.x = hashPoints[t.v2_indx].second; pts.push_back(pt);
		pt.y = -hashPoints[t.v3_indx].first; pt.x = hashPoints[t.v3_indx].second; pts.push_back(pt);
		cv::Point ptTemp1; ptTemp1.y = -hashPoints[t.v1_indx].first; ptTemp1.x = hashPoints[t.v1_indx].second;
		cv::Point ptTemp2; ptTemp2.y = -hashPoints[t.v2_indx].first; ptTemp2.x = hashPoints[t.v2_indx].second; 
		cv::Point ptTemp3; ptTemp3.y = -hashPoints[t.v3_indx].first; ptTemp3.x = hashPoints[t.v3_indx].second;
		cv::circle(imageOut, ptTemp1, 4, cv::Scalar(255, 0, 0));
		cv::circle(imageOut, ptTemp2, 4, cv::Scalar(0, 255, 0));
		cv::circle(imageOut, ptTemp3, 4, cv::Scalar(0, 0, 255));
		cv::fillPoly(imageOut, pts, cv::Scalar(255/20 * i, 255 / 20 * i, 255 / 20 * i));
	}
	cv::Point pt;
	pt.y = -hashPoints[indx + 1].first;
	pt.x = hashPoints[indx + 1].second;
	cv::circle(imageOut, pt, 2, cv::Scalar(255, 255, 255));
	 
	imwrite(".\\standard_test_images\\standard_test_images\\output" + std::to_string(indx) + ".png", imageOut);

}

struct listNode {

	TreeNode* incTri1;
	TreeNode* incTri2;

	int pk;
	int pl;
	int pi;
	int pj;
};

int vectorDir(Vec2 a) {
	double dot = a.y * 1.0;
	if (abs(dot - 0.0) < EPSILON) return 0;
	if (dot > 0.0) return 1; return -1;
}

struct classComparePoint {
	bool operator() (const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const{
		if (lhs.first != rhs.first) return lhs.first > rhs.first;
		return lhs.second > rhs.second;
	}
};

TreeNode* searchForTriOnLine(TreeNode* root, int pi, int pj, const std::vector<std::pair<int, int>>& hashPoints) {

	for (int i = 0; i < root->adjSiblings.size(); ++i) {
		if (root->adjSiblings[i]->isInternal == false) {
			TreeNode* adjSib = root->adjSiblings[i];
			int vertixCount = 0;
			if (root->triangle.v1_indx == adjSib->triangle.v1_indx ||
				root->triangle.v1_indx == adjSib->triangle.v2_indx ||
				root->triangle.v1_indx == adjSib->triangle.v3_indx) vertixCount++;
			if (root->triangle.v2_indx == adjSib->triangle.v1_indx ||
				root->triangle.v2_indx == adjSib->triangle.v2_indx ||
				root->triangle.v2_indx == adjSib->triangle.v3_indx) vertixCount++;
			if (root->triangle.v3_indx == adjSib->triangle.v1_indx ||
				root->triangle.v3_indx == adjSib->triangle.v2_indx ||
				root->triangle.v3_indx == adjSib->triangle.v3_indx) vertixCount++;
			//is an adj tri
			//the adj triangle also needs for root to be added
			if (vertixCount == 2) return adjSib;
		}
	}

	for (int i = 0; i < root->adjList.size(); ++i) {
		if (root->adjList[i]->isInternal == false) {
			TreeNode* adj = root->adjList[i];
			int vertixCount = 0;
			if (root->triangle.v1_indx == adj->triangle.v1_indx ||
				root->triangle.v1_indx == adj->triangle.v2_indx ||
				root->triangle.v1_indx == adj->triangle.v3_indx) vertixCount++;
			if (root->triangle.v2_indx == adj->triangle.v1_indx ||
				root->triangle.v2_indx == adj->triangle.v2_indx ||
				root->triangle.v2_indx == adj->triangle.v3_indx) vertixCount++;
			if (root->triangle.v3_indx == adj->triangle.v1_indx ||
				root->triangle.v3_indx == adj->triangle.v2_indx ||
				root->triangle.v3_indx == adj->triangle.v3_indx) vertixCount++;
			//is an adj tri
			//the adj triangle also needs for root to be added
			if (vertixCount == 2) return adj;
		}
	}
}

void addAdjTriangle(TreeNode* root, const std::vector<std::pair<int, int>>& hashPoints) {

	for (int pCounter = 0; pCounter < root->parent.size(); ++pCounter) {
		TreeNode* parent = root->parent[pCounter];
		for (int i = 0; i < parent->adjList.size(); ++i) {
			if (parent->adjList[i]->isInternal == false) {
				int vertixCount = 0;
				if (root->triangle.v1_indx == parent->adjList[i]->triangle.v1_indx ||
					root->triangle.v1_indx == parent->adjList[i]->triangle.v2_indx ||
					root->triangle.v1_indx == parent->adjList[i]->triangle.v3_indx) vertixCount++;
				if (root->triangle.v2_indx == parent->adjList[i]->triangle.v1_indx ||
					root->triangle.v2_indx == parent->adjList[i]->triangle.v2_indx ||
					root->triangle.v2_indx == parent->adjList[i]->triangle.v3_indx) vertixCount++;
				if (root->triangle.v3_indx == parent->adjList[i]->triangle.v1_indx ||
					root->triangle.v3_indx == parent->adjList[i]->triangle.v2_indx ||
					root->triangle.v3_indx == parent->adjList[i]->triangle.v3_indx) vertixCount++;
				//is an adj tri
				//the adj triangle also needs for root to be added
				if (vertixCount == 2) {
					root->adjList.push_back(parent->adjList[i]);
					parent->adjList[i]->adjList.push_back(root);
				}
			}
		}
	}

	for (int pCounter = 0; pCounter < root->parent.size(); ++pCounter) {
		TreeNode* parent = root->parent[pCounter];
		for (int i = 0; i < parent->adjSiblings.size(); ++i) {
			if (parent->adjSiblings[i]->isInternal == false) {
				int vertixCount = 0;
				if (root->triangle.v1_indx == parent->adjSiblings[i]->triangle.v1_indx ||
					root->triangle.v1_indx == parent->adjSiblings[i]->triangle.v2_indx ||
					root->triangle.v1_indx == parent->adjSiblings[i]->triangle.v3_indx) vertixCount++;
				if (root->triangle.v2_indx == parent->adjSiblings[i]->triangle.v1_indx ||
					root->triangle.v2_indx == parent->adjSiblings[i]->triangle.v2_indx ||
					root->triangle.v2_indx == parent->adjSiblings[i]->triangle.v3_indx) vertixCount++;
				if (root->triangle.v3_indx == parent->adjSiblings[i]->triangle.v1_indx ||
					root->triangle.v3_indx == parent->adjSiblings[i]->triangle.v2_indx ||
					root->triangle.v3_indx == parent->adjSiblings[i]->triangle.v3_indx) vertixCount++;
				//is an adj tri
				//the adj triangle also needs for root to be added
				if (vertixCount == 2) {
					root->adjList.push_back(parent->adjSiblings[i]);
					parent->adjSiblings[i]->adjList.push_back(root);
				}
			}
		}
	}
}

int calculate2DlineSideTest(int pi, int pj, int pr, const std::vector<std::pair<int, int>>& hashPoints) {
	//return 1 if pr is on left side of vector pi -> pj
	//return -1 if on right
	//return 0 if pr is on line pi -> pj
	if ((pi == 0 && pj == -1) || (pi == -2 && pj == 0) || (pi == -1 && pj == -2)) return -1;
	if ((pj == 0 && pi == -1) || (pj == -2 && pi == 0) || (pj == -1 && pi == -2)) return 1;

	if (pi == -1 ){
		bool isLexHigher = (hashPoints[pr].first > hashPoints[pj].first) ? true :
			(hashPoints[pr].first == hashPoints[pj].first && hashPoints[pr].second < hashPoints[pj].first) ? true : false;
		if (isLexHigher) return -1; return 1;
	}
	if (pj == -1) {
		bool isLexHigher = (hashPoints[pr].first > hashPoints[pi].first) ? true :
			(hashPoints[pr].first == hashPoints[pi].first && hashPoints[pr].second < hashPoints[pi].first) ? true : false;
		if (isLexHigher) return 1; return -1;
	}
	if (pi == -2) {
		bool isLexHigher = (hashPoints[pr].first > hashPoints[pj].first) ? true :
			(hashPoints[pr].first == hashPoints[pj].first && hashPoints[pr].second < hashPoints[pj].first) ? true : false;
		if (isLexHigher) return 1; return -1;
	}
	if (pj == -2) {
		bool isLexHigher = (hashPoints[pr].first > hashPoints[pi].first) ? true :
			(hashPoints[pr].first == hashPoints[pi].first && hashPoints[pr].second < hashPoints[pi].first) ? true : false;
		if (isLexHigher) return -1; return 1;
	}

	Vec2 tempVec(hashPoints[pj].second - hashPoints[pi].second, hashPoints[pj].first - hashPoints[pi].first);
	Vec2 tempPr(hashPoints[pr].second - hashPoints[pi].second, hashPoints[pr].first -  hashPoints[pi].first);
	double tempDeter = tempVec.x * tempPr.y - tempVec.y * tempPr.x;
	if (abs(tempDeter - 0.0) < EPSILON) return 0;
	if (tempDeter < 0.0) return -1; return 1;
}

std::list<listNode> searchForTriangles(TreeNode* root, int pr_indx, const std::vector<std::pair<int, int>>& hashPoints) {

	std::list<listNode > returnList;
	std::queue<TreeNode* > q;
	q.push(root);
	while ( !q.empty()) {
		auto top = q.front();
		int side1 = calculate2DlineSideTest(top->triangle.v1_indx, top->triangle.v2_indx, pr_indx, hashPoints);
		int side2 = calculate2DlineSideTest(top->triangle.v2_indx, top->triangle.v3_indx, pr_indx, hashPoints);
		int side3 = calculate2DlineSideTest(top->triangle.v3_indx, top->triangle.v1_indx, pr_indx, hashPoints);
		//point lies on a line of the triangle, must find the other triangle that contains the point
		if ((side1 == 0 && side2 == side3) || (side2 == 0 && side1 == side3 ) || ( side3 == 0 && side2 == side1)) {
			if (top->isInternal) {
				for (int i = 0; i < LINKS_SIZE; ++i) { if (top->links[i] != NULL) q.push(top->links[i]); }
			}
			else {
				if (side1 == 0) {
					TreeNode* adjTri = searchForTriOnLine(top, top->triangle.v1_indx, top->triangle.v2_indx, hashPoints);
					listNode tempListNode; tempListNode.incTri1 = top; tempListNode.pk = top->triangle.v3_indx;
					tempListNode.pi = top->triangle.v1_indx; tempListNode.pj = top->triangle.v2_indx;
					listNode tempListNodeAdj; tempListNodeAdj.incTri2 = adjTri; 
					returnList.push_back(tempListNode); returnList.push_back(tempListNodeAdj);
				}
				else if (side2 == 0) {
					TreeNode* adjTri = searchForTriOnLine(top, top->triangle.v2_indx, top->triangle.v3_indx, hashPoints);
					listNode tempListNode; tempListNode.incTri1 = top; tempListNode.pk = top->triangle.v1_indx;
					tempListNode.pi = top->triangle.v2_indx; tempListNode.pj = top->triangle.v3_indx;
					listNode tempListNodeAdj; tempListNodeAdj.incTri2 = adjTri;
					returnList.push_back(tempListNode); returnList.push_back(tempListNodeAdj);
				}
				else {
					TreeNode* adjTri = searchForTriOnLine(top, top->triangle.v3_indx, top->triangle.v1_indx, hashPoints);
					listNode tempListNode; tempListNode.incTri1 = top; tempListNode.pk = top->triangle.v2_indx;
					tempListNode.pi = top->triangle.v3_indx; tempListNode.pj = top->triangle.v1_indx;
					listNode tempListNodeAdj; tempListNodeAdj.incTri2 = adjTri;
					returnList.push_back(tempListNode); returnList.push_back(tempListNodeAdj);
				}
			}
			return returnList;
		}
		//does not lie on the line of the triangle
		//is the point contained within the triangle?
		else if (side1 == side2 && side1 == side3 && side3 == side2) {
			if (top->isInternal) {
				for (int i = 0; i < LINKS_SIZE; ++i) { if (top->links[i] != NULL) q.push(top->links[i]); }
			}
			else { listNode tempListNode; tempListNode.incTri1 = top; returnList.push_back(tempListNode); return returnList; }
		}
		//else throw "Point did not land in any triangle.";
		q.pop();
	}
	throw "Did not find a triangle";
	return returnList;
}

void legalizeEdge(int pr_indx, int pi, int pj, TreeNode* root, const std::vector<std::pair<int, int>>& hashPoints) {

	//all these edges are legal
	if ((pi == -1 && pj == -2)|| (pi == -2 && pj == -1) ||
		(pi == -1 && pj == 0) || (pi == 0 && pj == -1) ||
		(pi == -2 && pj == 0) || (pi == 0 && pj == -2)) return;
	
	int pk = 0;
	for (int i = 0; i < root->adjList.size(); ++i) {
		TreeNode* adjNode = root->adjList[i];
		if (adjNode->isInternal == true) continue;
		const Triangle & adjTri = adjNode->triangle;
		if ((pi == adjTri.v1_indx && pj == adjTri.v2_indx) || (pi == adjTri.v2_indx && pj == adjTri.v1_indx)) pk = adjTri.v3_indx;
		else if ((pi == adjTri.v1_indx && pj == adjTri.v3_indx) || (pi == adjTri.v3_indx && pj == adjTri.v1_indx)) pk = adjTri.v2_indx;
		else if ((pi == adjTri.v2_indx && pj == adjTri.v3_indx) || (pi == adjTri.v3_indx && pj == adjTri.v2_indx)) pk = adjTri.v1_indx;
		//this adj tri is not even incident to line pi pj
		else continue;
		bool isEdgeLegal = true ;
		if (pi < 0 || pj < 0 || pk < 0)
			isEdgeLegal = min(pi, pj) > pk;
		else {
			Circle c = createCircle(root->triangle, hashPoints); Vec2 center = c.c; Vec2 pk_loc(hashPoints[pk].second, hashPoints[pk].first);
			isEdgeLegal = distanceEqnt(center, pk_loc) >= c.r;
		}
		//edge pi -> pj or pj -> pi is not a legal edge
		//split triangle, create two new triangles
		//if pi or pj == INF vertex, then the edge might not be flipped necessarily.
		//if pi or pj == INF vertex then the edge may only be flipped if pi and pj lie on opposite sides of the newly flipped edge
		//if they both lie on the same side of the newly flipped edge then the convex hull is illigitemate and the edge is legal
		//if they are both on opposite sides of the newly flipped edge then its a legitamte flip
		if (!isEdgeLegal) {
			std::cout << "Illegal edge detected" << std::endl;
			if (pi == -1 || pi == -2 || pj == -1 || pj == -2) {
				if (pk == -1 || pk == -2) return;
				Vec2 a(hashPoints[pk].second - hashPoints[pr_indx].second, hashPoints[pk].first - hashPoints[pr_indx].first);
				int prpkDir = vectorDir(a);
				if (prpkDir == 0) {
					int horizDir = a.x;
					
					if ((pj == -1 && horizDir < 0 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != -1) ||
						(pi == -1 && horizDir < 0 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != -1) ||
						(pj == -1 && horizDir > 0 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != 1) ||
						(pi == -1 && horizDir > 0 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != 1) ||
						(pj == -2 && horizDir < 0 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != 1) ||
						(pi == -2 && horizDir < 0 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != 1) ||
						(pj == -2 && horizDir > 0 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != -1) ||
						(pi == -2 && horizDir > 0 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != -1)) {
						return;
					}
				}
				else {
					if ((pj == -1 && prpkDir == -1 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != -1) ||
						(pi == -1 && prpkDir == -1 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != -1) ||
						(pj == -1 && prpkDir == 1 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != 1) ||
						(pi == -1 && prpkDir == 1 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != 1) ||
						(pj == -2 && prpkDir == -1 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != 1) ||
						(pi == -2 && prpkDir == -1 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != 1) ||
						(pj == -2 && prpkDir == 1 && calculate2DlineSideTest(pr_indx, pk, pi, hashPoints) != -1) ||
						(pi == -2 && prpkDir == 1 && calculate2DlineSideTest(pr_indx, pk, pj, hashPoints) != -1)) {
						return;
					}
				}
			}
			TreeNode* tri1 = new TreeNode(pr_indx, pi, pk, root); tri1->pushParent(adjNode);
			TreeNode* tri2 = new TreeNode(pr_indx, pj, pk, root); tri2->pushParent(adjNode);
			tri1->adjSiblings.push_back(tri2); tri2->adjSiblings.push_back(tri1);
			root->isInternal = true; adjNode->isInternal = true;
			root->links[0] = adjNode->links[0] = tri1;
			root->links[1] = adjNode->links[1] = tri2;
			addAdjTriangle(tri1, hashPoints); addAdjTriangle(tri2, hashPoints);
			legalizeEdge(pr_indx, pi, pk, tri1, hashPoints);
			legalizeEdge(pr_indx, pk, pj, tri2, hashPoints);
			return;
		}
	}
}


void colorImage(TreeNode* root, cv::Mat imgO, const std::vector<std::pair<int, int>>& hashPoints) {

	cv::Mat imageOut(imgO.rows, imgO.cols, CV_8UC3);
	std::unordered_set<TreeNode*> leafNodes;
	std::queue<TreeNode* > q;
	q.push(root);
	while (!q.empty()) {
		auto top = q.front();
		if (top->isInternal == true) {
			for (int i = 0; i < LINKS_SIZE; ++i) {
				if (top->links[i] != NULL)
					q.push(top->links[i]);
			}
		}
		else leafNodes.insert(top);
		q.pop();
	}

	for (auto& v : leafNodes) {
		if (v->triangle.v1_indx != -1 && v->triangle.v1_indx != -2 &&
			v->triangle.v2_indx != -1 && v->triangle.v2_indx != -2 &&
			v->triangle.v3_indx != -1 && v->triangle.v3_indx != -2) {
			cv::Vec3b coloredPoint1, coloredPoint2, coloredPoint3;
			cv::Point pt1; pt1.x = hashPoints[v->triangle.v1_indx].second; pt1.y = -hashPoints[v->triangle.v1_indx].first;
			cv::Point pt2; pt2.x = hashPoints[v->triangle.v2_indx].second; pt2.y = -hashPoints[v->triangle.v2_indx].first;
			cv::Point pt3; pt3.x = hashPoints[v->triangle.v3_indx].second; pt3.y = -hashPoints[v->triangle.v3_indx].first;
			coloredPoint1 = imgO.at<cv::Vec3b>(pt1); coloredPoint2 = imgO.at<cv::Vec3b>(pt2); coloredPoint3 = imgO.at<cv::Vec3b>(pt3);
			v->color[0] = (coloredPoint1[0] + coloredPoint2[0] + coloredPoint3[0]) / 3;
			v->color[1] = (coloredPoint1[1] + coloredPoint2[1] + coloredPoint3[1]) / 3;
			v->color[2] = (coloredPoint1[2] + coloredPoint2[2] + coloredPoint3[2]) / 3;
		}
	}

	for (int i = 0; i < imageOut.rows; ++i) {
		for (int j = 0; j < imageOut.cols; ++j) {
			cv::Point pt; pt.x = i; pt.y = j;
			cv::Vec3b inputColor; inputColor[0] = inputColor[1] = inputColor[2] = 255;
			imageOut.at<cv::Vec3b>(pt) = inputColor;
			for (auto& v : leafNodes) {
				if (v->triangle.v1_indx != -1 && v->triangle.v1_indx != -2 &&
					v->triangle.v2_indx != -1 && v->triangle.v2_indx != -2 &&
					v->triangle.v3_indx != -1 && v->triangle.v3_indx != -2) {

					int side1 = 0, side2 = 0, side3 = 0;
					Vec2 tempVec(hashPoints[v->triangle.v2_indx].second - hashPoints[v->triangle.v1_indx].second,
						-hashPoints[v->triangle.v2_indx].first + hashPoints[v->triangle.v1_indx].first);
					Vec2 tempPr( pt.x - hashPoints[v->triangle.v1_indx].second, pt.y + hashPoints[v->triangle.v1_indx].first);
					double tempDeter = tempVec.x * tempPr.y - tempVec.y * tempPr.x;
					if (abs(tempDeter - 0.0) < EPSILON) side1 = 0;
					if (tempDeter < 0.0) side1 = -1; else side1 = 1;

					tempVec.x = hashPoints[v->triangle.v3_indx].second - hashPoints[v->triangle.v2_indx].second;
					tempVec.y = -hashPoints[v->triangle.v3_indx].first + hashPoints[v->triangle.v2_indx].first;
					tempPr.x = pt.x - hashPoints[v->triangle.v2_indx].second;
					tempPr.y = pt.y + hashPoints[v->triangle.v2_indx].first;
					tempDeter = tempVec.x * tempPr.y - tempVec.y * tempPr.x;
					if (abs(tempDeter - 0.0) < EPSILON) side2 = 0;
					if (tempDeter < 0.0) side2 = -1; else side2 = 1;

					tempVec.x = hashPoints[v->triangle.v1_indx].second - hashPoints[v->triangle.v3_indx].second;
					tempVec.y = -hashPoints[v->triangle.v1_indx].first + hashPoints[v->triangle.v3_indx].first;
					tempPr.x = pt.x - hashPoints[v->triangle.v3_indx].second;
					tempPr.y = pt.y + hashPoints[v->triangle.v3_indx].first;
					tempDeter = tempVec.x * tempPr.y - tempVec.y * tempPr.x;
					if (abs(tempDeter - 0.0) < EPSILON) side3 = 0;
					if (tempDeter < 0.0) side3 = -1; else side3 = 1;

					if ((side1 == side2 && side1 == side3 && side3 == side2) || ( side1 == 0 || side2 == 0 || side3 == 0) ) {
						double uu, vv;
						extractUV(uu, vv,pt,  v,  hashPoints);
						cv::Point pt1; pt1.x = hashPoints[v->triangle.v1_indx].second; pt1.y = -hashPoints[v->triangle.v1_indx].first;
						cv::Point pt2; pt2.x = hashPoints[v->triangle.v2_indx].second; pt2.y = -hashPoints[v->triangle.v2_indx].first;
						cv::Point pt3; pt3.x = hashPoints[v->triangle.v3_indx].second; pt3.y = -hashPoints[v->triangle.v3_indx].first;
						cv::Point finalPt;
						/*
						if (pt1.y > pt2.y) {
							if (pt1.y > pt3.y) {
								finalPt.x = pt1.x; finalPt.y = pt1.y;
							}
							else {
								finalPt.x = pt3.x; finalPt.y = pt3.y;
							}
						}
						else {
							if (pt2.y > pt3.y) {
								finalPt.x = pt2.x; finalPt.y = pt2.y;
							}
							else {
								finalPt.x = pt3.x; finalPt.y = pt3.y;
							}
						}
						*/
						
						for (int ii = 0; ii < 3; ++ii) {
							
							inputColor[ii] = uu * (imgO.at<cv::Vec3b>(pt1))[ii] +
								vv * (imgO.at<cv::Vec3b>(pt2))[ii] +
								(1.0 - uu - vv) * (imgO.at<cv::Vec3b>(pt3))[ii];
						}
						
						//inputColor[0] = (imgO.at<cv::Vec3b>(finalPt))[0]; inputColor[1] = (imgO.at<cv::Vec3b>(finalPt))[1];
						//inputColor[2] = (imgO.at<cv::Vec3b>(finalPt))[2];
						//inputColor = imageOut.at<cv::Vec3b>(finalPt);
						imageOut.at<cv::Vec3b>(pt) = inputColor;
						break;
					}
				}
			}
		}
	}
	imwrite(".\\standard_test_images\\standard_test_images\\outputFinal.png", imageOut);
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
	//swap index 0 in hashPoints for p0
	std::pair<int, int> tempPair = hashPoints[0];
	hashPoints[0] = hashPoints[p0Indx];
	hashPoints[p0Indx] = tempPair;

	TreeNode* root = new TreeNode(0, -1, -2, NULL);

	for (int i = 1; i < hashPoints.size(); ++i) {
		
		//search for the triangle in the tree that contains your point and is a leaf
		//two triangles may contain your point if the point lies on a line
		std::list<listNode> triList = searchForTriangles(root, i, hashPoints);
		//is contained in only one triangle
		//this creates 3 new triangles
		if (triList.size() == 1) {
			auto top = triList.front();
			TreeNode* tri1 = new TreeNode(i, top.incTri1->triangle.v1_indx, top.incTri1->triangle.v2_indx, top.incTri1);
			TreeNode* tri2 = new TreeNode(i, top.incTri1->triangle.v2_indx, top.incTri1->triangle.v3_indx, top.incTri1);
			TreeNode* tri3 = new TreeNode(i, top.incTri1->triangle.v3_indx, top.incTri1->triangle.v1_indx, top.incTri1);
			top.incTri1->isInternal = true;
			top.incTri1->links[0] = tri1; top.incTri1->links[1] = tri2; top.incTri1->links[2] = tri3;
			tri1->adjSiblings.push_back(tri2); tri1->adjSiblings.push_back(tri3);
			tri2->adjSiblings.push_back(tri1); tri2->adjSiblings.push_back(tri3);
			tri3->adjSiblings.push_back(tri2); tri3->adjSiblings.push_back(tri1);
			//check for other adj triangles
			//each triangle added must check for adj triangles with the adj list of it's parent
			addAdjTriangle(tri1, hashPoints); addAdjTriangle(tri2, hashPoints); addAdjTriangle(tri3, hashPoints);
			legalizeEdge(i, tri1->triangle.v2_indx, tri1->triangle.v3_indx, tri1, hashPoints);
			legalizeEdge(i, tri2->triangle.v2_indx, tri2->triangle.v3_indx, tri2, hashPoints);
			legalizeEdge(i, tri3->triangle.v2_indx, tri3->triangle.v3_indx, tri3, hashPoints);
		}
		//is contained on the line, so is contained in two triangles
		//this creates 4 new triangles
		else {
			std::cout << "Landed on line" << std::endl;
			TreeNode* triInc1 = triList.front().incTri1; TreeNode* triInc2 = triList.back().incTri2;
			int pk = triList.front().pk; int pi = triList.front().pi; int pj = triList.front().pj;
			TreeNode* tri1 = new TreeNode(i, pi, pk, triInc1);
			TreeNode* tri2 = new TreeNode(i, pj, pk, triInc1);
			triInc1->isInternal = true;
			triInc1->links[0] = tri1; triInc1->links[1] = tri2;
			tri1->adjSiblings.push_back(tri2); tri2->adjSiblings.push_back(tri1);
			int pl = 0;
			if ((pi == triInc2->triangle.v1_indx && pj == triInc2->triangle.v2_indx) ||
				(pi == triInc2->triangle.v2_indx && pj == triInc2->triangle.v1_indx)) pl = triInc2->triangle.v3_indx;
			else if ((pi == triInc2->triangle.v1_indx && pj == triInc2->triangle.v3_indx) || 
					    (pi == triInc2->triangle.v3_indx && pj == triInc2->triangle.v1_indx)) pl = triInc2->triangle.v2_indx;
			else if ((pi == triInc2->triangle.v2_indx && pj == triInc2->triangle.v3_indx) ||
					    (pi == triInc2->triangle.v3_indx && pj == triInc2->triangle.v2_indx)) pl = triInc2->triangle.v1_indx;
			TreeNode* tri3 = new TreeNode(i, pi, pl, triInc2);
			TreeNode* tri4 = new TreeNode(i, pj, pl, triInc2);
			triInc2->isInternal = true;
			triInc2->links[0] = tri3; triInc2->links[1] = tri4;
			tri3->adjSiblings.push_back(tri4); tri4->adjSiblings.push_back(tri3);
			tri1->adjList.push_back(tri3); tri3->adjList.push_back(tri1);
			tri2->adjList.push_back(tri4); tri4->adjList.push_back(tri2);
			addAdjTriangle(tri1, hashPoints); addAdjTriangle(tri2, hashPoints); addAdjTriangle(tri3, hashPoints); addAdjTriangle(tri4, hashPoints);
			legalizeEdge(i, pi, pk, tri1, hashPoints);
			legalizeEdge(i, pj, pk, tri2, hashPoints);
			legalizeEdge(i, pi, pl, tri3, hashPoints);
			legalizeEdge(i, pj, pl, tri4, hashPoints);
		}
		//triangleImage(root, image, i, hashPoints);
		colorImage(root, image, hashPoints);
		//createDOTFile(root, hashPoints, i);
	}
}
