#include <iostream>
#include <list>
#include <unordered_map>
#include <memory>
#include <atomic>
#pragma once


struct BakeryStats {
    std::atomic<int> eggs_laid{0};
    std::atomic<int> eggs_used{0};
    std::atomic<int> butter_produced{0};
    std::atomic<int> butter_used{0};
    std::atomic<int> sugar_produced{0};
    std::atomic<int> sugar_used{0};
    std::atomic<int> flour_produced{0};
    std::atomic<int> flour_used{0};
    std::atomic<int> cakes_produced{0};
    std::atomic<int> cakes_sold{0};

	void print() const {
        std::cout
          << "\n\n\n\n\n\nBakeryStats:\n"
          << "  eggs_laid:        " << eggs_laid.load()       << "\n"
          << "  eggs_used:        " << eggs_used.load()       << "\n"
          << "  butter_produced:  " << butter_produced.load() << "\n"
          << "  butter_used:      " << butter_used.load()     << "\n"
          << "  sugar_produced:   " << sugar_produced.load()  << "\n"
          << "  sugar_used:       " << sugar_used.load()      << "\n"
          << "  flour_produced:   " << flour_produced.load()  << "\n"
          << "  flour_used:       " << flour_used.load()      << "\n"
          << "  cakes_produced:   " << cakes_produced.load()  << "\n"
          << "  cakes_sold:       " << cakes_sold.load()      << "\n";
    }
};

class DisplayObject {
public:

	//DO NOT CHANGE THE TYPES OR NAMES OF THESE VARIABLES
	int  width;
	int  height;
	int  layer;
	int  x;
	int  y;
	int  id;
	std::string texture;
	int  direction;  

	void setPos(int, int);
	void setTexture(const std::string&);
	void setDirection(int);

	DisplayObject(const std::string&, const int, const int, const int, const int);
	~DisplayObject();
	void updateFarm();
	void erase();

	static void redisplay(BakeryStats& stats);

	//DO NOT CHANGE WIDTH AND HEIGHT
	static const int WIDTH = 800;
	static const int HEIGHT = 600;

	static std::unordered_map<int, DisplayObject> theFarm;
	static BakeryStats stats;


	//DO NOT CHANGE THE TYPE OF THIS VARIABLE
	static std::shared_ptr<std::unordered_map<int, DisplayObject>> buffedFarmPointer;
	
private:
	
};
