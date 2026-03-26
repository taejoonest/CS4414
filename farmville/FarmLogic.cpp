#include "FarmLogic.h"
#include "displayobject.hpp"
#include <unistd.h>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>
#include <deque>
#include <map>
#include <condition_variable>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cugl/core/math/CURect.h>
#include <cugl/core/math/CUVec2.h>




std::mutex farmMutex;  
std::atomic<bool> running(true);
BakeryStats* globalStats;


template<typename T, const int LEN>
class RingBuffer {
private:
    std::mutex mtx;
    std::condition_variable not_full, not_empty;
    std::deque<T> buffer;

public:
    void produce(const T& obj) {
        std::unique_lock<std::mutex> plock(mtx);
        not_full.wait(plock, [this]() { return buffer.size() < LEN || !running; });
        if (!running) return;
        buffer.push_back(obj);
        not_empty.notify_one();
    }


    bool try_produce(const T& obj) {
        std::scoped_lock lock(mtx); 
        if (buffer.size() >= LEN || !running) return false;
        buffer.push_back(obj);
        not_empty.notify_one();
        return true;
    }

    bool consume(T& obj) {
        std::unique_lock<std::mutex> clock(mtx);
        not_empty.wait(clock, [this]() { return !buffer.empty() || !running; });
        if (!running || buffer.empty()) return false;
        obj = buffer.front();
        buffer.pop_front();
        not_full.notify_one();
        return true;
    }


    bool try_consume(T& obj) {
        std::scoped_lock lock(mtx);  
        if (buffer.empty() || !running) return false;
        obj = buffer.front();
        buffer.pop_front();
        not_full.notify_one();
        return true;
    }

    int size() {
        std::scoped_lock lock(mtx);
        return buffer.size();
    }
};


class NestMonitor {
private:
    int x, y;
    RingBuffer<int, 3> eggBuffer; 
    std::atomic<bool> chickenOccupied;
    std::mutex chickenIdMutex;
    std::vector<int> chickenIdsUsed;  

public:
    NestMonitor(int xPos, int yPos) : x(xPos), y(yPos), chickenOccupied(false) {}

    int getX() { return x; }
    int getY() { return y; }

  
    bool tryClaimNest() {
        return !chickenOccupied.exchange(true);  
    }

    void releaseNest() {
        chickenOccupied.store(false);
    }

    bool hasChickenUsedNest(int chickenId) {
        std::scoped_lock lock(chickenIdMutex);
        for (int id : chickenIdsUsed) {
            if (id == chickenId) return true;
        }
        return false;
    }

    void markChickenUsed(int chickenId) {
        std::scoped_lock lock(chickenIdMutex);
        chickenIdsUsed.push_back(chickenId);
    }

    bool tryLayEgg(DisplayObject* nestEggs[], int nestId) {
    
        if (eggBuffer.try_produce(1)) {
            globalStats->eggs_laid++;
            int eggIndex = eggBuffer.size() - 1;
            
 
            
                std::scoped_lock flock(farmMutex);
            nestEggs[nestId * 3 + eggIndex]->updateFarm();
            
            return true;
        }
        return false; 
    }

    int farmerCollect(DisplayObject* nestEggs[], int nestId) {
       
        if (chickenOccupied.load()) {
            return 0;  
        }
        
        int collected = 0;
        int egg;
        
        
        while (eggBuffer.try_consume(egg)) {
            collected++;
        }
       
        if (collected > 0) {
            
            std::scoped_lock flock(farmMutex);
            for (int i = 0; i < collected; i++) {
            nestEggs[nestId * 3 + i]->erase();
        }
        
        }
        
        
        if (eggBuffer.size() == 0) {
            std::scoped_lock lock(chickenIdMutex);
            chickenIdsUsed.clear();
        }
        
        return collected;
    }

    int getEggCount() {
        return eggBuffer.size();
    }
};


class BarnMonitor {
private:
    RingBuffer<int, 12>* eggBuffer;  
    bool hasEggBuffer;
    std::mutex mutex;

public:
    BarnMonitor(bool withEggBuffer) : hasEggBuffer(withEggBuffer) {
        if (hasEggBuffer) {
            eggBuffer = new RingBuffer<int, 12>();  
        } else {
            eggBuffer = nullptr;
        }
    }

    ~BarnMonitor() {
        if (eggBuffer) delete eggBuffer;
    }

    void deliverEggs(int count) {
    
        if (hasEggBuffer && eggBuffer) {
            for (int i = 0; i < count; i++) {
                if (!eggBuffer->try_produce(1)) {
                    break;  
                }
            }
        }
    }

    void loadTruck(int& eggs, int& otherProduct) {
        if (hasEggBuffer && eggBuffer) {
            
            if (eggBuffer->size() >= 3) {
                eggs = 0;
                int egg;
                for (int i = 0; i < 3; i++) {
                    if (eggBuffer->try_consume(egg)) {  
                        eggs++;
                    } else {
                        break;  
                    }
                }
                otherProduct = 3;  
                globalStats->butter_produced += 3;
            } else {
                
                eggs = 0;
                otherProduct = 0;
            }
        } else {
           
            eggs = 0;
            otherProduct = 3;  
            globalStats->flour_produced += 3;
            globalStats->sugar_produced += 3;
        }
    }
    
    void emptyEggs() {
        if (hasEggBuffer && eggBuffer) {
            int egg;
            while (eggBuffer->try_consume(egg)) {
                
            }
        }
    }
};


class BakeryStorageMonitor {
private:
    RingBuffer<int, 6>* eggBuffer;
    RingBuffer<int, 6>* butterBuffer;
    RingBuffer<int, 6>* flourBuffer;
    RingBuffer<int, 6>* sugarBuffer;
    DisplayObject** eggDisplay;
    DisplayObject** butterDisplay;
    DisplayObject** flourDisplay;
    DisplayObject** sugarDisplay;
    DisplayObject** travelEggs;
    DisplayObject** travelButter;
    DisplayObject** travelFlour;
    DisplayObject** travelSugar;
    std::mutex mutex;
    
    
    int eggFixedPositions[6][2];
    int butterFixedPositions[6][2];
    int flourFixedPositions[6][2];
    int sugarFixedPositions[6][2];
    
   
    bool animating;
    int animationFrame;

public:
    BakeryStorageMonitor(DisplayObject* eggs[], DisplayObject* butter[], 
                         DisplayObject* flour[], DisplayObject* sugar[],
                         DisplayObject* travelEgg[], DisplayObject* travelBut[],
                         DisplayObject* travelFlo[], DisplayObject* travelSug[],
                         int stockBaseX, int stockBaseY, int spacing) {
        eggBuffer = new RingBuffer<int, 6>();
        butterBuffer = new RingBuffer<int, 6>();
        flourBuffer = new RingBuffer<int, 6>();
        sugarBuffer = new RingBuffer<int, 6>();
        eggDisplay = eggs;
        butterDisplay = butter;
        flourDisplay = flour;
        sugarDisplay = sugar;
        travelEggs = travelEgg;
        travelButter = travelBut;
        travelFlour = travelFlo;
        travelSugar = travelSug;
        
        
        for (int i = 0; i < 6; i++) {
            eggFixedPositions[i][0] = stockBaseX + i * spacing;
            eggFixedPositions[i][1] = stockBaseY;
            
            butterFixedPositions[i][0] = stockBaseX + i * spacing;
            butterFixedPositions[i][1] = stockBaseY + spacing;
            
            flourFixedPositions[i][0] = stockBaseX + i * spacing;
            flourFixedPositions[i][1] = stockBaseY + spacing * 2;
            
            sugarFixedPositions[i][0] = stockBaseX + i * spacing;
            sugarFixedPositions[i][1] = stockBaseY + spacing * 3;
        }
        
        animating = false;
        animationFrame = 0;
    }

    ~BakeryStorageMonitor() {
        delete eggBuffer;
        delete butterBuffer;
        delete flourBuffer;
        delete sugarBuffer;
    }

   
    bool tryUnloadOne(const std::string& ingredient) {
  
        if (animating) {
            return false;
        }
        
        std::scoped_lock lock(farmMutex);  
        
        if (ingredient == "egg") {
            int currentCount = eggBuffer->size();
            if (eggBuffer->try_produce(1)) {
                if (currentCount < 6) {
                    eggDisplay[currentCount]->updateFarm();
                }
                return true;
            }
        } else if (ingredient == "butter") {
            int currentCount = butterBuffer->size();
            if (butterBuffer->try_produce(1)) {
                if (currentCount < 6) {
                    butterDisplay[currentCount]->updateFarm();
                }
                return true;
            }
        } else if (ingredient == "flour") {
            int currentCount = flourBuffer->size();
            if (flourBuffer->try_produce(1)) {
                if (currentCount < 6) {
                    flourDisplay[currentCount]->updateFarm();
                }
                return true;
            }
        } else if (ingredient == "sugar") {
            int currentCount = sugarBuffer->size();
            if (sugarBuffer->try_produce(1)) {
                if (currentCount < 6) {
                    sugarDisplay[currentCount]->updateFarm();
                }
                return true;
            }
        }
        return false; 
    }

    void unloadTruck(int eggs, int butter, int flour, int sugar) {
       
        std::scoped_lock lock(farmMutex);  
        
        int currentEggCount = eggBuffer->size();
        for (int i = 0; i < eggs; i++) {
            if (eggBuffer->try_produce(1)) {
                if (currentEggCount < 6) {
                    eggDisplay[currentEggCount]->updateFarm();
                    currentEggCount++;
                }
            } else break;
        }
        
        int currentButterCount = butterBuffer->size();
        for (int i = 0; i < butter; i++) {
            if (butterBuffer->try_produce(1)) {
                if (currentButterCount < 6) {
                    butterDisplay[currentButterCount]->updateFarm();
                    currentButterCount++;
                }
            } else break;
        }
        
        int currentFlourCount = flourBuffer->size();
        for (int i = 0; i < flour; i++) {
            if (flourBuffer->try_produce(1)) {
                if (currentFlourCount < 6) {
                    flourDisplay[currentFlourCount]->updateFarm();
                    currentFlourCount++;
                }
            } else break;
        }
        
        int currentSugarCount = sugarBuffer->size();
        for (int i = 0; i < sugar; i++) {
            if (sugarBuffer->try_produce(1)) {
                if (currentSugarCount < 6) {
                    sugarDisplay[currentSugarCount]->updateFarm();
                    currentSugarCount++;
                }
            } else break;
        }
    }

    bool takeIngredients() {
  
        if (eggBuffer->size() >= 2 && butterBuffer->size() >= 2 &&
            flourBuffer->size() >= 2 && sugarBuffer->size() >= 2) {
            
            int item;
            int eggsGot = 0, butterGot = 0, flourGot = 0, sugarGot = 0;
            
  
            for (int i = 0; i < 2; i++) {
                if (eggBuffer->try_consume(item)) eggsGot++;
                else break;
            }
            for (int i = 0; i < 2; i++) {
                if (butterBuffer->try_consume(item)) butterGot++;
                else break;
            }
            for (int i = 0; i < 2; i++) {
                if (flourBuffer->try_consume(item)) flourGot++;
                else break;
            }
            for (int i = 0; i < 2; i++) {
                if (sugarBuffer->try_consume(item)) sugarGot++;
                else break;
            }
            

            if (eggsGot == 2 && butterGot == 2 && flourGot == 2 && sugarGot == 2) {
           
                std::scoped_lock lock(farmMutex);
                
                int newEggCount = eggBuffer->size();
                int newButterCount = butterBuffer->size();
                int newFlourCount = flourBuffer->size();
                int newSugarCount = sugarBuffer->size();
                
            
                for (int i = newEggCount; i < newEggCount + 2; i++) {
                    if (i < 6) eggDisplay[i]->erase();
                }
                for (int i = newButterCount; i < newButterCount + 2; i++) {
                    if (i < 6) butterDisplay[i]->erase();
                }
                for (int i = newFlourCount; i < newFlourCount + 2; i++) {
                    if (i < 6) flourDisplay[i]->erase();
                }
                for (int i = newSugarCount; i < newSugarCount + 2; i++) {
                    if (i < 6) sugarDisplay[i]->erase();
                }
            
            globalStats->eggs_used += 2;
            globalStats->butter_used += 2;
            globalStats->flour_used += 2;
            globalStats->sugar_used += 2;
            return true;
            }

        }
        return false;
    }
    
    bool isAnimating() {
        return animating;
    }
    
    bool startTakeIngredients() {
        if (animating) return false;
        
     
        if (eggBuffer->size() < 2 || butterBuffer->size() < 2 ||
            flourBuffer->size() < 2 || sugarBuffer->size() < 2) {
            return false;
        }
        
     
        int item;
        int eggsGot = 0, butterGot = 0, flourGot = 0, sugarGot = 0;
        
        for (int i = 0; i < 2; i++) {
            if (eggBuffer->try_consume(item)) eggsGot++;
            else break;
        }
        for (int i = 0; i < 2; i++) {
            if (butterBuffer->try_consume(item)) butterGot++;
            else break;
        }
        for (int i = 0; i < 2; i++) {
            if (flourBuffer->try_consume(item)) flourGot++;
            else break;
        }
        for (int i = 0; i < 2; i++) {
            if (sugarBuffer->try_consume(item)) sugarGot++;
            else break;
        }
        
       
        if (eggsGot != 2 || butterGot != 2 || flourGot != 2 || sugarGot != 2) {
            return false;
        }
        
     
        std::scoped_lock lock(farmMutex);
        
        
        travelEggs[0]->setPos(eggFixedPositions[0][0], eggFixedPositions[0][1]);
        travelEggs[1]->setPos(eggFixedPositions[1][0], eggFixedPositions[1][1]);
        travelButter[0]->setPos(butterFixedPositions[0][0], butterFixedPositions[0][1]);
        travelButter[1]->setPos(butterFixedPositions[1][0], butterFixedPositions[1][1]);
        travelFlour[0]->setPos(flourFixedPositions[0][0], flourFixedPositions[0][1]);
        travelFlour[1]->setPos(flourFixedPositions[1][0], flourFixedPositions[1][1]);
        travelSugar[0]->setPos(sugarFixedPositions[0][0], sugarFixedPositions[0][1]);
        travelSugar[1]->setPos(sugarFixedPositions[1][0], sugarFixedPositions[1][1]);
        
        
        travelEggs[0]->updateFarm();
        travelEggs[1]->updateFarm();
        travelButter[0]->updateFarm();
        travelButter[1]->updateFarm();
        travelFlour[0]->updateFarm();
        travelFlour[1]->updateFarm();
        travelSugar[0]->updateFarm();
        travelSugar[1]->updateFarm();
        
    
        eggDisplay[0]->erase();
        eggDisplay[1]->erase();
        butterDisplay[0]->erase();
        butterDisplay[1]->erase();
        flourDisplay[0]->erase();
        flourDisplay[1]->erase();
        sugarDisplay[0]->erase();
        sugarDisplay[1]->erase();
        
        animating = true;
        animationFrame = 0;
        
        globalStats->eggs_used += 2;
        globalStats->butter_used += 2;
        globalStats->flour_used += 2;
        globalStats->sugar_used += 2;
        
        return true;
    }
    
    void updateAnimation() {
        if (!animating) return;
        
        std::scoped_lock lock(farmMutex);
        animationFrame++;
        
        if (animationFrame <= 12) {
            
            float deltaY = animationFrame * (70.0f / 12.0f);
            
            travelEggs[0]->setPos(eggFixedPositions[0][0], eggFixedPositions[0][1] - (int)deltaY);
            travelEggs[1]->setPos(eggFixedPositions[1][0], eggFixedPositions[1][1] - (int)deltaY);
            travelButter[0]->setPos(butterFixedPositions[0][0], butterFixedPositions[0][1] - (int)deltaY);
            travelButter[1]->setPos(butterFixedPositions[1][0], butterFixedPositions[1][1] - (int)deltaY);
            travelFlour[0]->setPos(flourFixedPositions[0][0], flourFixedPositions[0][1] - (int)deltaY);
            travelFlour[1]->setPos(flourFixedPositions[1][0], flourFixedPositions[1][1] - (int)deltaY);
            travelSugar[0]->setPos(sugarFixedPositions[0][0], sugarFixedPositions[0][1] - (int)deltaY);
            travelSugar[1]->setPos(sugarFixedPositions[1][0], sugarFixedPositions[1][1] - (int)deltaY);
            
            travelEggs[0]->updateFarm();
            travelEggs[1]->updateFarm();
            travelButter[0]->updateFarm();
            travelButter[1]->updateFarm();
            travelFlour[0]->updateFarm();
            travelFlour[1]->updateFarm();
            travelSugar[0]->updateFarm();
            travelSugar[1]->updateFarm();
            
        } else if (animationFrame <= 24) {
           
            float progress = (animationFrame - 12) / 12.0f;  
            int deltaX = (int)(progress * 70);  
            
           
            travelEggs[0]->setPos(eggFixedPositions[0][0] + deltaX, eggFixedPositions[0][1] - 70);
            travelEggs[1]->setPos(eggFixedPositions[1][0] + deltaX, eggFixedPositions[1][1] - 70);
            travelButter[0]->setPos(butterFixedPositions[0][0] + deltaX, butterFixedPositions[0][1] - 70);
            travelButter[1]->setPos(butterFixedPositions[1][0] + deltaX, butterFixedPositions[1][1] - 70);
            travelFlour[0]->setPos(flourFixedPositions[0][0] + deltaX, flourFixedPositions[0][1] - 70);
            travelFlour[1]->setPos(flourFixedPositions[1][0] + deltaX, flourFixedPositions[1][1] - 70);
            travelSugar[0]->setPos(sugarFixedPositions[0][0] + deltaX, sugarFixedPositions[0][1] - 70);
            travelSugar[1]->setPos(sugarFixedPositions[1][0] + deltaX, sugarFixedPositions[1][1] - 70);
            
            travelEggs[0]->updateFarm();
            travelEggs[1]->updateFarm();
            travelButter[0]->updateFarm();
            travelButter[1]->updateFarm();
            travelFlour[0]->updateFarm();
            travelFlour[1]->updateFarm();
            travelSugar[0]->updateFarm();
            travelSugar[1]->updateFarm();
            
        } else if (animationFrame == 25) {
          
            travelEggs[0]->erase();
            travelEggs[1]->erase();
            travelButter[0]->erase();
            travelButter[1]->erase();
            travelFlour[0]->erase();
            travelFlour[1]->erase();
            travelSugar[0]->erase();
            travelSugar[1]->erase();
            
        } else if (animationFrame <= 37) {
            
            float progress = (animationFrame - 25) / 12.0f;  
            
            
            for (int i = 2; i <= 5; i++) {
                if (i < eggBuffer->size() + 2) {  
                    int startX = eggFixedPositions[i][0];
                    int endX = eggFixedPositions[i-2][0];
                    int currentX = startX + (int)((endX - startX) * progress);
                    
                    eggDisplay[i]->setPos(currentX, eggFixedPositions[i][1]);
                    eggDisplay[i]->updateFarm();
                }
            }
            
          
            for (int i = 2; i <= 5; i++) {
                if (i < butterBuffer->size() + 2) {
                    int startX = butterFixedPositions[i][0];
                    int endX = butterFixedPositions[i-2][0];
                    int currentX = startX + (int)((endX - startX) * progress);
                    
                    butterDisplay[i]->setPos(currentX, butterFixedPositions[i][1]);
                    butterDisplay[i]->updateFarm();
                }
            }
            
          
            for (int i = 2; i <= 5; i++) {
                if (i < flourBuffer->size() + 2) {
                    int startX = flourFixedPositions[i][0];
                    int endX = flourFixedPositions[i-2][0];
                    int currentX = startX + (int)((endX - startX) * progress);
                    
                    flourDisplay[i]->setPos(currentX, flourFixedPositions[i][1]);
                    flourDisplay[i]->updateFarm();
                }
            }
            
           
            for (int i = 2; i <= 5; i++) {
                if (i < sugarBuffer->size() + 2) {
                    int startX = sugarFixedPositions[i][0];
                    int endX = sugarFixedPositions[i-2][0];
                    int currentX = startX + (int)((endX - startX) * progress);
                    
                    sugarDisplay[i]->setPos(currentX, sugarFixedPositions[i][1]);
                    sugarDisplay[i]->updateFarm();
                }
            }
            
    } else if (animationFrame == 38) {
       
        int newEggCount = eggBuffer->size();
        int newButterCount = butterBuffer->size();
        int newFlourCount = flourBuffer->size();
        int newSugarCount = sugarBuffer->size();
        
       
        for (int i = 0; i < 6; i++) {
         
            eggDisplay[i]->setPos(eggFixedPositions[i][0], eggFixedPositions[i][1]);
            if (i < newEggCount) {
                eggDisplay[i]->updateFarm();
            } else {
                eggDisplay[i]->erase();
            }
            
            
            butterDisplay[i]->setPos(butterFixedPositions[i][0], butterFixedPositions[i][1]);
            if (i < newButterCount) {
                butterDisplay[i]->updateFarm();
            } else {
                butterDisplay[i]->erase();
            }
            
           
            flourDisplay[i]->setPos(flourFixedPositions[i][0], flourFixedPositions[i][1]);
            if (i < newFlourCount) {
                flourDisplay[i]->updateFarm();
            } else {
                flourDisplay[i]->erase();
            }
            
        
            sugarDisplay[i]->setPos(sugarFixedPositions[i][0], sugarFixedPositions[i][1]);
            if (i < newSugarCount) {
                sugarDisplay[i]->updateFarm();
            } else {
                sugarDisplay[i]->erase();
            }
        }
        
        animating = false;
        animationFrame = 0;
    }
    }
};


struct SharedQueueState {
    std::mutex mutex;
    int childPositions[5];  
    const int lineSpacing = 50; 
    
    SharedQueueState() {

        for (int i = 0; i < 5; i++) {
            childPositions[i] = i;
        }
    }
    

    int getPosition(int childId) {
        int childIndex = childId - 9; 
        std::scoped_lock lock(mutex);
        return childPositions[childIndex];
    }
    
 
    void childFinishedShopping(int childId) {
        int childIndex = childId - 9;
        std::scoped_lock lock(mutex);
        
    
        for (int i = 0; i < 5; i++) {
            if (i != childIndex && childPositions[i] > 0) {
                childPositions[i]--;
            }
        }
        
       
        childPositions[childIndex] = 4;
    }
};


class BakeryShopMonitor {
private:
    RingBuffer<int, 6>* cakeBuffer;
    bool shopOccupied;
    std::mutex mutex;
    DisplayObject** cakeDisplay;
    int framesSinceBaked;  

public:
    BakeryShopMonitor(DisplayObject* cakes[]) {
        cakeBuffer = new RingBuffer<int, 6>();
        shopOccupied = false;
        cakeDisplay = cakes;
        framesSinceBaked = 0;
    }

    ~BakeryShopMonitor() {
        delete cakeBuffer;
    }

    void addCakes(int count) {
        for (int i = 0; i < count; i++) {
            cakeBuffer->produce(1);
        }
        globalStats->cakes_produced += count;
        framesSinceBaked = 1;  
    }

    bool enterShop() {
        
        std::unique_lock<std::mutex> lock(mutex);
        
        if (!shopOccupied) {  
            shopOccupied = true;
            
            return true;
        }
       
        return false;
    }

    void leaveShop() {
        std::scoped_lock lock(mutex);
        shopOccupied = false;
    }

    void incrementFrameCounter() {
   
        if (framesSinceBaked > 0 && framesSinceBaked < 7) {
            framesSinceBaked++;
        } else if (framesSinceBaked >= 7) {
            framesSinceBaked = 0; 
        }
    }

    int buyCakes(int requested) {
      
        if (framesSinceBaked > 0 && framesSinceBaked < 7) {
            return 0; 
        }
        
 
        int bought = 0;
        int cake;
        for (int i = 0; i < requested; i++) {
            if (cakeBuffer->try_consume(cake)) {
                bought++;
            } else {
                break;  
            }
        }
        if (bought > 0) {
            globalStats->cakes_sold += bought;
            
            std::scoped_lock lock(farmMutex);
            int remaining = cakeBuffer->size();
            for (int i = 0; i < 6; i++) {
                if (i < remaining) {
                    cakeDisplay[i]->updateFarm();
                } else {
                    cakeDisplay[i]->erase();
                }
            }
        }
        return bought;
    }

    int getCakeCount() {
        return cakeBuffer->size();
    }
};

class UpdateBarrier {
private:
    std::mutex mutex;
    std::condition_variable threadsDone;        
    std::condition_variable frameReady;         
    std::condition_variable proposalsDone;      
    std::condition_variable collisionsDone;     
    int threadsFinished = 0;
    int proposalsFinished = 0;
    int totalThreads;
    int currentEpoch = 0;        
    std::vector<bool> collisionReady; 
    std::vector<bool> threadCompletedFrame;  
    
public:
    UpdateBarrier(int total) : totalThreads(total), collisionReady(total, false), threadCompletedFrame(total, false) {}
    
 
    void proposalSubmitted() {
        std::unique_lock<std::mutex> lock(mutex);
        proposalsFinished++;
        
        if (proposalsFinished == totalThreads) {
            
            proposalsDone.notify_one();  
        }
    }
    

    void waitForAllProposals() {
        
        std::unique_lock<std::mutex> lock(mutex);
        proposalsDone.wait(lock, [this]{ return proposalsFinished == totalThreads || !running; });
       
    }
    
 
    void collisionsChecked() {
        std::unique_lock<std::mutex> lock(mutex);
   
        for (int i = 0; i < totalThreads; i++) {
            collisionReady[i] = true;
        }
        
        collisionsDone.notify_all(); 
    }
    

    void waitForCollisionResults(int threadId) {
       
        std::unique_lock<std::mutex> lock(mutex);
    
        collisionsDone.wait(lock, [this, threadId]{ 
            return collisionReady[threadId] || !running; 
        });
       
    }
    

    void threadFinished(int threadId) {
        std::unique_lock<std::mutex> lock(mutex);
        
 
        if (threadId >= 0 && threadId < totalThreads) {
            threadCompletedFrame[threadId] = true;
        }
        

        threadsFinished++;
       
        int myEpoch = currentEpoch;  
        
        if (threadsFinished == totalThreads) {
            
            threadsDone.notify_one();
        }
        
 
        frameReady.wait(lock, [this, myEpoch]{ return currentEpoch > myEpoch || !running; });
    }
    

    void waitForAllThreads() {
        std::unique_lock<std::mutex> lock(mutex);
        
    
        while (threadsFinished < totalThreads && running) {
            if (threadsDone.wait_for(lock, std::chrono::seconds(3)) == std::cv_status::timeout) {
             
                for (int i = 0; i < totalThreads; i++) {
                    if (!threadCompletedFrame[i]) {
                        std::cout << i << " ";
                    }
                }
                std::cout << std::endl;
          
            }
        }
    }
    

    void frameComplete() {
        std::unique_lock<std::mutex> lock(mutex);
        threadsFinished = 0;
        proposalsFinished = 0;
 
        for (int i = 0; i < totalThreads; i++) {
            collisionReady[i] = false;
            threadCompletedFrame[i] = false;
        }
        currentEpoch++;  
        frameReady.notify_all();  
    }
};


struct ProposedMove {
    DisplayObject* object;
    int proposedX;
    int proposedY;
    bool isValid;  
    
    ProposedMove() : object(nullptr), proposedX(0), proposedY(0), isValid(false) {}
};

class Layer2CollisionMonitor {
private:
    std::vector<ProposedMove> proposals;
    std::vector<bool> hasCollision;
    std::vector<int> blockedByID;
    std::vector<int> blockerXPos;
    std::vector<int> blockerYPos;
    std::mutex proposalMutex;
    int numObjects;

  
    cugl::Rect getRect(int x, int y, float width, float height) {
        return cugl::Rect(
            x - width / 2.0f,
            y - height / 2.0f,
            width,
            height
        );
    }

  
    cugl::Rect getRectAtPosition(DisplayObject* obj, int x, int y) {
        return getRect(x, y, obj->width, obj->height);
    }

   
    bool checkCollision(cugl::Rect r1, cugl::Rect r2) {
        return r1.doesIntersect(r2);
    }

public:
    Layer2CollisionMonitor(int numMovingObjects) : numObjects(numMovingObjects) {
        proposals.resize(numMovingObjects);
        hasCollision.resize(numMovingObjects);
        blockedByID.resize(numMovingObjects);
        blockerXPos.resize(numMovingObjects);
        blockerYPos.resize(numMovingObjects);
    }
    
   
    void proposeMove(int threadIdx, DisplayObject* obj, int newX, int newY, bool valid) {
        if (threadIdx >= 0 && threadIdx < numObjects) {
            proposals[threadIdx].object = obj;
            proposals[threadIdx].proposedX = newX;
            proposals[threadIdx].proposedY = newY;
            proposals[threadIdx].isValid = valid;
        }
    }
    
  
    void checkAllCollisions() {
        
    
        for (int i = 0; i < numObjects; i++) {
            hasCollision[i] = false;
            blockedByID[i] = -1;
            blockerXPos[i] = 0;
            blockerYPos[i] = 0;
        }
        
    
        for (int i = 0; i < numObjects; i++) {
            if (!proposals[i].isValid) continue;
            
            cugl::Rect rectI = getRectAtPosition(
                proposals[i].object, 
                proposals[i].proposedX, 
                proposals[i].proposedY
            );
            
            for (int j = i + 1; j < numObjects; j++) {
                if (!proposals[j].isValid) continue;
                
                cugl::Rect rectJ = getRectAtPosition(
                    proposals[j].object,
                    proposals[j].proposedX,
                    proposals[j].proposedY
                );
                
                if (checkCollision(rectI, rectJ)) {
                
                    int idI = proposals[i].object->id;
                    int idJ = proposals[j].object->id;
                    
              
                    bool bothChickens = (idI >= 0 && idI <= 3) && (idJ >= 0 && idJ <= 3);
                    
                    if (bothChickens) {
                        
                        int vxI = proposals[i].proposedX - proposals[i].object->x;
                        int vyI = proposals[i].proposedY - proposals[i].object->y;
                        int vxJ = proposals[j].proposedX - proposals[j].object->x;
                        int vyJ = proposals[j].proposedY - proposals[j].object->y;
                        
                       
                        bool iStationary = (vxI == 0 && vyI == 0);
                        bool jStationary = (vxJ == 0 && vyJ == 0);
                        
                        bool futureCollision = false;
                        
                    
                        if (!iStationary && !jStationary) {
                       
                            const int PREDICT_FRAMES = 10;
                            for (int frame = 1; frame <= PREDICT_FRAMES; frame++) {
                                int futureXI = proposals[i].proposedX + vxI * frame;
                                int futureYI = proposals[i].proposedY + vyI * frame;
                                int futureXJ = proposals[j].proposedX + vxJ * frame;
                                int futureYJ = proposals[j].proposedY + vyJ * frame;
                                
                            
                                cugl::Rect futureRectI = getRectAtPosition(proposals[i].object, futureXI, futureYI);
                                cugl::Rect futureRectJ = getRectAtPosition(proposals[j].object, futureXJ, futureYJ);
                                if (checkCollision(futureRectI, futureRectJ)) {
                                    futureCollision = true;
                break;
                                }
                            }
                        }
                
                        
                        
                        
                        if (futureCollision) {

                            hasCollision[i] = true;
                            hasCollision[j] = true;
                        
                            blockedByID[i] = idJ;  
                            blockedByID[j] = idI;  
                   
                            blockerXPos[i] = -1;  
                            blockerYPos[i] = -1;
                            blockerXPos[j] = -1;
                            blockerYPos[j] = -1;
            } else {
                            
                            if (idI < idJ) {
                           
                                hasCollision[j] = true;
                                blockedByID[j] = -3;  
                                blockerXPos[j] = proposals[i].proposedX;
                                blockerYPos[j] = proposals[i].proposedY;
                            } else {
                             
                                hasCollision[i] = true;
                                blockedByID[i] = -3;  
                                blockerXPos[i] = proposals[j].proposedX;
                                blockerYPos[i] = proposals[j].proposedY;
            }
        }
                    } else {
                       
                        
                        
                        if (idI < idJ) {
                        
                            hasCollision[j] = true;
                            blockedByID[j] = idI;
                            blockerXPos[j] = proposals[i].proposedX;
                            blockerYPos[j] = proposals[i].proposedY;
            } else {
                         
                            hasCollision[i] = true;
                            blockedByID[i] = idJ;
                            blockerXPos[i] = proposals[j].proposedX;
                            blockerYPos[i] = proposals[j].proposedY;
                        }
                    }
                }
            }
        }
       
    }
    

    bool wasBlocked(int threadIdx, int& blockedBy, int& blockerX, int& blockerY) {
        if (threadIdx >= 0 && threadIdx < numObjects) {
            if (hasCollision[threadIdx]) {
                blockedBy = blockedByID[threadIdx];
                blockerX = blockerXPos[threadIdx];
                blockerY = blockerYPos[threadIdx];
        return true;
    }
        }
        blockedBy = -1;
        return false;
    }
};


struct TruckState {
    std::mutex mutex;
    float posX, posY;        
    float velocityX, velocityY;  
    bool goingToStorage;    
    
    TruckState() : posX(0), posY(0), velocityX(0), velocityY(0), goingToStorage(false) {}
};


static TruckState truckState6;
static TruckState truckState7;


struct PredictedCollision {
    std::mutex mutex;
    bool collisionPredicted;
    int truckToStop;  
    
    PredictedCollision() : collisionPredicted(false), truckToStop(-1) {}
};
static PredictedCollision predictedCollision;



void redisplayThread(BakeryStats* stats, UpdateBarrier* barrier, Layer2CollisionMonitor* collisionMonitor, DisplayObject* nestObjects[], BakeryShopMonitor* shop, BarnMonitor* barn1) {
    int frameCount = 0;
    const int EMPTY_INTERVAL = 450;  
    
    while (running) {
        
       
        if (frameCount > 0 && frameCount % EMPTY_INTERVAL == 0) {
            barn1->emptyEggs();
        }
       
        shop->incrementFrameCounter();
        
    
        barrier->waitForAllProposals();
        
  
      
        collisionMonitor->checkAllCollisions();
        
     
        barrier->collisionsChecked();
        
  
      
        barrier->waitForAllThreads();
        
      
        {
            std::scoped_lock lock(farmMutex);
            nestObjects[0]->updateFarm();
            nestObjects[1]->updateFarm();
            nestObjects[2]->updateFarm();
        }
        

        
        {
            std::scoped_lock lock(farmMutex);
            DisplayObject::redisplay(*stats);
        }
        

       
        barrier->frameComplete();
        
        frameCount++;
   
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
}

void chickenThread(DisplayObject* chicken, int chickenId, 
                   std::vector<NestMonitor*>& nests,
                   DisplayObject* nestObjects[],
                   DisplayObject* nestEggs[],
                   Layer2CollisionMonitor* collisionMonitor,
                   std::vector<DisplayObject*>& allLayer2,
                   UpdateBarrier* barrier) {
   
    
    int eggsLaidSinceMove = 0;
    int currentTargetNestIdx = -1; 
    int lastTimedOutNestIdx = -1;  
    int lastUsedNestIdx = -1;  
    bool isLayingEggs = false;
    int layingStartTime = 0;
    int eggsToLayThisVisit = 0;  
    int waitingAtNestFrames = 0;  
    int chickenSpeed = 1 + (std::rand() % 2);  
    
    while (running) {

        if (currentTargetNestIdx == -1 && !isLayingEggs) {
         
            chickenSpeed = 1 + (std::rand() % 2);
            
            std::vector<int> availableNests;
            for (size_t i = 0; i < nests.size(); i++) {
                if (nests[i]->getEggCount() < 3 && (int)i != lastTimedOutNestIdx && (int)i != lastUsedNestIdx) {
                    availableNests.push_back(i);
                }
            }
            

            if (availableNests.empty()) {
                for (size_t i = 0; i < nests.size(); i++) {
                    if (nests[i]->getEggCount() < 3) {
                        availableNests.push_back(i);
                    }
                }
                lastTimedOutNestIdx = -1;
                lastUsedNestIdx = -1; 
            }
            
     
            if (!availableNests.empty()) {
                int randomIdx = std::rand() % availableNests.size();
                currentTargetNestIdx = availableNests[randomIdx];
                waitingAtNestFrames = 0;  
            } else {
      

                currentTargetNestIdx = std::rand() % nests.size();
                waitingAtNestFrames = 0;  
                lastTimedOutNestIdx = -1;
                lastUsedNestIdx = -1; 
            }
        }
        
        NestMonitor* targetNest = nests[currentTargetNestIdx];
        int targetX = targetNest->getX();
        int targetY = targetNest->getY();
        
    
        int dx = 0, dy = 0;
        int proposedX = chicken->x;
        int proposedY = chicken->y;
        bool shouldMove = !isLayingEggs;
        
        if (shouldMove) {
        if (chicken->x < targetX - 5) dx = chickenSpeed;
        else if (chicken->x > targetX + 5) dx = -chickenSpeed;
        
        if (chicken->y < targetY - 5) dy = chickenSpeed;
        else if (chicken->y > targetY + 5) dy = -chickenSpeed;
        
            proposedX = chicken->x + dx;
            proposedY = chicken->y + dy;
        
      
            if (proposedX < 30) proposedX = 30;
            if (proposedX > 770) proposedX = 770;
            if (proposedY < 350) proposedY = 350;
            if (proposedY > 570) proposedY = 570;
        }
        

        collisionMonitor->proposeMove(chickenId, chicken, proposedX, proposedY, shouldMove);
        barrier->proposalSubmitted();
        
       
        barrier->waitForCollisionResults(chickenId);
        

        int blockedBy = -1, blockerX = 0, blockerY = 0;
        
        if (!shouldMove) {
     
            {
                std::scoped_lock lock(farmMutex);
                chicken->updateFarm();
            }
        }
        
        if (shouldMove) {
            if (!collisionMonitor->wasBlocked(chickenId, blockedBy, blockerX, blockerY)) {
                
               
                chicken->setPos(proposedX, proposedY);
                
               
                if (dx < 0) {
                    chicken->setDirection(1);   
                } else if (dx > 0) {
                    chicken->setDirection(-1);  
                }
                
                {
                    
                std::scoped_lock lock(farmMutex);
                chicken->updateFarm();
                    
                }
            } else {
                
                
                if (blockedBy == -3) {
                   
                   
                    
                    {
                        std::scoped_lock lock(farmMutex);
                        chicken->updateFarm();
                    }
                } else if (blockerX == -1 && blockerY == -1) {
                    
                    
                    int altX = chicken->x;
                    int altY = chicken->y;
                    
                    
                    int detourDistance = chickenSpeed * 5;  
                    if (chickenId < blockedBy) {
                     
                        if (dx == 0 && dy == 0) {
                       
                            altY = chicken->y + detourDistance;
                        } else if (dx == 0) {
                         
                            altX = chicken->x + detourDistance;
                        } else {
                           
                            altY = chicken->y + detourDistance;
                        }
                    } else {
                      
                        if (dx == 0 && dy == 0) {
                          
                            altY = chicken->y - detourDistance;
                        } else if (dx == 0) {
                      
                            altX = chicken->x - detourDistance;
                        } else {
                      
                            altY = chicken->y - detourDistance;
                        }
                    }
                    
                   
                    if (altX < 30) altX = 30;
                    if (altX > 770) altX = 770;
                    if (altY < 350) altY = 350;
                    if (altY > 570) altY = 570;
                    
                    chicken->setPos(altX, altY);
                    {
                        std::scoped_lock lock(farmMutex);
                        chicken->updateFarm();
                    }
                } else {
                   
                   
                    
                    int altX = chicken->x;
                    int altY = chicken->y;
                    int detourDistance = chickenSpeed * 5;  
                    
                    if (dx == 0) {
                     
                        altX = chicken->x + (chicken->x < blockerX ? -detourDistance : detourDistance);
                    } else {
                   
                        altY = chicken->y + (chicken->y < blockerY ? -detourDistance : detourDistance);
                    }
                    
                  
                    if (altX < 30) altX = 30;
                    if (altX > 770) altX = 770;
                    if (altY < 350) altY = 350;
                    if (altY > 570) altY = 570;
                    
                   
                    
                    chicken->setPos(altX, altY);
                    {
                        
                        std::scoped_lock lock(farmMutex);
                        chicken->updateFarm();
                        
                    }
                }
        }
        
  
        int distX = abs(chicken->x - targetX);
        int distY = abs(chicken->y - targetY);
            if (distX < 20 && distY < 20 && !isLayingEggs) {
                
                if (targetNest->getEggCount() >= 3) {
                  
                    waitingAtNestFrames++;
                    if (waitingAtNestFrames > 10) {
                        
                        lastTimedOutNestIdx = currentTargetNestIdx;  
                        currentTargetNestIdx = -1;
                        waitingAtNestFrames = 0;
                    }
                } else {
                    
                    if (currentTargetNestIdx == 2 && targetNest->hasChickenUsedNest(chickenId)) {
                      
                        lastTimedOutNestIdx = currentTargetNestIdx;
                        currentTargetNestIdx = -1;
                        waitingAtNestFrames = 0;
                    } else if (targetNest->tryClaimNest()) {
                      
                        if (targetNest->getEggCount() >= 3) {
                            
                            targetNest->releaseNest();
                            currentTargetNestIdx = -1;
                            waitingAtNestFrames = 0;
                        } else {
                        
                            isLayingEggs = true;
                            layingStartTime = 0;
                eggsLaidSinceMove = 0;
                         
                            if (currentTargetNestIdx == 2) {
                                eggsToLayThisVisit = 1 + (std::rand() % 2); 
                            } else {
                                eggsToLayThisVisit = 1 + (std::rand() % 3);  
                            }
                            waitingAtNestFrames = 0;  
                        }
                    }
                   
                  
                }
            }
        }
        
        
        if (isLayingEggs) {
            layingStartTime++;
            

            if (layingStartTime % 3 == 0 && layingStartTime >= 9 && eggsLaidSinceMove < eggsToLayThisVisit) {
                
                if (targetNest->getEggCount() >= 3) {
                    
                    if (currentTargetNestIdx == 2 && eggsLaidSinceMove > 0) {
                        targetNest->markChickenUsed(chickenId);
                    }
                    
                    targetNest->releaseNest();
                    isLayingEggs = false;
                    currentTargetNestIdx = -1;  
                } else {
             
                    if (targetNest->tryLayEgg(nestEggs, currentTargetNestIdx)) {
                       
            eggsLaidSinceMove++;
                    }
                  
                    
                    if (eggsLaidSinceMove >= eggsToLayThisVisit) {
                    
                       
                        if (currentTargetNestIdx == 2) {
                            targetNest->markChickenUsed(chickenId);
                        }
                        
                        targetNest->releaseNest();
                        isLayingEggs = false;
                        lastUsedNestIdx = currentTargetNestIdx;  
                        currentTargetNestIdx = -1;  
                    }
                }
            }
            if (layingStartTime > 100) {
               
                if (currentTargetNestIdx == 2 && eggsLaidSinceMove > 0) {
                    targetNest->markChickenUsed(chickenId);
                }
                
                targetNest->releaseNest();
                isLayingEggs = false;
                currentTargetNestIdx = -1;  
            }
        }
        
       
        barrier->threadFinished(chickenId);
    }
}

void farmerThread(DisplayObject* farmer,
                  int farmerId,
                  std::vector<NestMonitor*>& nests,
                  DisplayObject* nestEggs[],
                  BarnMonitor* barn1,
                  Layer2CollisionMonitor* collisionMonitor,
                  std::vector<DisplayObject*>& allLayer2,
                  UpdateBarrier* barrier) {
    const int barn1X = 50, barn1Y = 50;
    const int farmerY = 530;  
    const int leftPathX = 30;  
    
    int currentNestIdx = -1;  
    int eggsCollected = 0;
    
    enum FarmerState {
        AT_NEST_LEVEL,      
        TO_LEFT_PATH,       
        TO_BARN,            
        FROM_BARN          
    };
    
    FarmerState state = AT_NEST_LEVEL;
    
    while (running) {
        int targetX, targetY;
        
        if (state == TO_LEFT_PATH) {
          
            targetX = leftPathX;
            targetY = farmerY;
        } else if (state == TO_BARN) {
         
            targetX = barn1X;
            targetY = barn1Y;
        } else if (state == FROM_BARN) {
          
            targetX = leftPathX;
            targetY = farmerY;
        } else {
           
            if (currentNestIdx == -1) {
            
                int maxEggs = 0;
                for (size_t i = 0; i < nests.size(); i++) {
                    int eggCount = nests[i]->getEggCount();
                    if (eggCount > maxEggs) {
                        maxEggs = eggCount;
                    }
                }
                
               
                if (maxEggs >= 1) {
                    std::vector<int> maxNests;
                    for (size_t i = 0; i < nests.size(); i++) {
                        if (nests[i]->getEggCount() == maxEggs) {
                            maxNests.push_back(i);
                        }
                    }
                
                    if (!maxNests.empty()) {
                        int randomIdx = std::rand() % maxNests.size();
                        currentNestIdx = maxNests[randomIdx];
                    }
                }
             
            }
            
            
            if (currentNestIdx == -1) {
                targetX = farmer->x;
                targetY = farmer->y;
        } else {
               
            targetX = nests[currentNestIdx]->getX();
                targetY = farmerY;  
            }
        }
        
        int dx = 0, dy = 0;
        int speed = 7; 
        if (farmer->x < targetX - 5) dx = speed;
        else if (farmer->x > targetX + 5) dx = -speed;
        
        if (farmer->y < targetY - 5) dy = speed;
        else if (farmer->y > targetY + 5) dy = -speed;
        
        int proposedX = farmer->x + dx;
        int proposedY = farmer->y + dy;
        
       
        if (proposedX < 15) proposedX = 15;
        if (proposedX > 785) proposedX = 785;
        if (proposedY < 30) proposedY = 30;
        if (proposedY > 570) proposedY = 570;
        
        bool shouldMove = (dx != 0 || dy != 0);
        collisionMonitor->proposeMove(farmerId, farmer, proposedX, proposedY, shouldMove);
        barrier->proposalSubmitted();
        
        
        barrier->waitForCollisionResults(farmerId);
        
       
        int blockedBy = -1, blockerX = 0, blockerY = 0;
        if (shouldMove) {
            if (!collisionMonitor->wasBlocked(farmerId, blockedBy, blockerX, blockerY)) {
               
                farmer->setPos(proposedX, proposedY);
            {
                std::scoped_lock lock(farmMutex);
                farmer->updateFarm();
            }
            } else {
               
                int altX = farmer->x;
                int altY = farmer->y;
                
                if (dx == 0) {
                    altX = farmer->x + (farmer->x < blockerX ? -20 : 20);
                } else {
                    altY = farmer->y + (farmer->y < blockerY ? -20 : 20);
                }
                
                if (altX < 15) altX = 15;
                if (altX > 785) altX = 785;
                if (altY < 30) altY = 30;
                if (altY > 570) altY = 570;
                
               
                farmer->setPos(altX, altY);
                {
                    std::scoped_lock lock(farmMutex);
                    farmer->updateFarm();
                }
            }
        }
        
        
        int distX = abs(farmer->x - targetX);
        int distY = abs(farmer->y - targetY);
        if (distX < 15 && distY < 15) {
            if (state == AT_NEST_LEVEL && currentNestIdx != -1) {
               
                int collected = nests[currentNestIdx]->farmerCollect(nestEggs, currentNestIdx);
                eggsCollected += collected;
                
                if (collected > 0) {
              
                    state = TO_LEFT_PATH;
                } else {
                 
                    currentNestIdx = -1;
                }
            } else if (state == TO_LEFT_PATH) {
       
                state = TO_BARN;
            } else if (state == TO_BARN) {
             
                if (eggsCollected > 0) {
                    barn1->deliverEggs(eggsCollected);
                    eggsCollected = 0;
                }
            
                state = FROM_BARN;
            } else if (state == FROM_BARN) {
               
                state = AT_NEST_LEVEL;
                currentNestIdx = -1; 
            }
        }
        
     
        barrier->threadFinished(farmerId);
    }
}

void truckThread(DisplayObject* truck, int truckId,
                 BarnMonitor* barn,
                 BakeryStorageMonitor* bakeryStorage,
                 Layer2CollisionMonitor* collisionMonitor,
                 std::vector<DisplayObject*>& allLayer2,
                 UpdateBarrier* barrier) {
  
    const int barn1X = 90, barn1Y = 50;       
    const int barn2X = 90, barn2Y = 150;       
    

    const int storageEggX = 525, storageEggY = 240;     
    const int storageFlourX = 525, storageFlourY = 180; 
    
    bool isEggTruck = (truckId == 6);
    

    bool goingToStorage = false;  
    
    int targetX = isEggTruck ? barn1X : barn2X;
    int targetY = isEggTruck ? barn1Y : barn2Y;
    bool hasLoad = false;
    int loadedEggs = 0, loadedButter = 0, loadedFlour = 0, loadedSugar = 0;
    
   
    float truckFloatX = static_cast<float>(truck->x);
    float truckFloatY = static_cast<float>(truck->y);
    
   
    bool stoppedForTruck = false;
    
    
    float speed = 4.0f;
    float moveX = 0.0f, moveY = 0.0f;
    
    
    float dirX = targetX - truckFloatX;
    float dirY = targetY - truckFloatY;
    float distance = std::sqrt(dirX * dirX + dirY * dirY);
    if (distance > 0.5f) {
        moveX = (dirX / distance) * speed;
        moveY = (dirY / distance) * speed;
    }
        
   
    TruckState& myState = (truckId == 6) ? truckState6 : truckState7;
    TruckState& otherState = (truckId == 6) ? truckState7 : truckState6;
    int otherTruckId = (truckId == 6) ? 7 : 6;
    
    while (running) {
      
        if (!goingToStorage) {
        
            float distX = std::abs(truckFloatX - (isEggTruck ? barn1X : barn2X));
            float distY = std::abs(truckFloatY - (isEggTruck ? barn1Y : barn2Y));
            
            if (distX < 5.0f && distY < 5.0f) {
             
                bool loadSuccessful = false;
                
                if (isEggTruck) {
                    barn->loadTruck(loadedEggs, loadedButter);
                    loadSuccessful = (loadedEggs > 0);  
                } else {
                    int dummy1, dummy2;
                    barn->loadTruck(dummy1, dummy2);  
                    loadedFlour = 3;
                    loadedSugar = 3;
                    loadedEggs = 0;   
                    loadedButter = 0; 
                    loadSuccessful = true; 
                }
                
                if (loadSuccessful) {
                hasLoad = true;
                    goingToStorage = true;
                    targetX = isEggTruck ? storageEggX : storageFlourX;
                    targetY = isEggTruck ? storageEggY : storageFlourY;
                    
                    
                    float dirX = targetX - truckFloatX;
                    float dirY = targetY - truckFloatY;
                    float distance = std::sqrt(dirX * dirX + dirY * dirY);
                    if (distance > 0.5f) {
                        moveX = (dirX / distance) * speed;
                        moveY = (dirY / distance) * speed;
                    }
                }
            }
            } else {
          
            float distX = std::abs(truckFloatX - (isEggTruck ? storageEggX : storageFlourX));
            float distY = std::abs(truckFloatY - (isEggTruck ? storageEggY : storageFlourY));
            
            if (distX < 5.0f && distY < 5.0f) {
               
                if (hasLoad) {
                    if (isEggTruck) {
                     
                        while (loadedEggs > 0) {
                            if (bakeryStorage->tryUnloadOne("egg")) {
                                loadedEggs--;
                    } else {
                                break; 
                            }
                        }
                     
                        while (loadedButter > 0) {
                            if (bakeryStorage->tryUnloadOne("butter")) {
                                loadedButter--;
                            } else {
                                break;  
                            }
                        }
                    } else {
                    
                        while (loadedFlour > 0) {
                            if (bakeryStorage->tryUnloadOne("flour")) {
                                loadedFlour--;
                            } else {
                                break;  
                            }
                        }
                    
                        while (loadedSugar > 0) {
                            if (bakeryStorage->tryUnloadOne("sugar")) {
                                loadedSugar--;
                            } else {
                                break;  
                            }
                        }
                    }
                    
               
                    if (loadedEggs == 0 && loadedButter == 0 && loadedFlour == 0 && loadedSugar == 0) {
                    hasLoad = false;
                     
                        goingToStorage = false;
                        targetX = isEggTruck ? barn1X : barn2X;
                        targetY = isEggTruck ? barn1Y : barn2Y;
                        
                        
                        float dirX = targetX - truckFloatX;
                        float dirY = targetY - truckFloatY;
                        float distance = std::sqrt(dirX * dirX + dirY * dirY);
                        if (distance > 0.5f) {
                            moveX = (dirX / distance) * speed;
                            moveY = (dirY / distance) * speed;
                        }
                    }
             
                }
            }
        }
        
     
        bool atStorageUnloading = goingToStorage && hasLoad && 
                                   std::abs(truckFloatX - (isEggTruck ? storageEggX : storageFlourX)) < 5.0f &&
                                   std::abs(truckFloatY - (isEggTruck ? storageEggY : storageFlourY)) < 5.0f;
        
     
        {
            std::scoped_lock lock(myState.mutex);
            myState.posX = truckFloatX;
            myState.posY = truckFloatY;
            myState.velocityX = moveX;
            myState.velocityY = moveY;
            myState.goingToStorage = goingToStorage;
        }
        
   
        {
           
            float otherPosX, otherPosY, otherVelX, otherVelY;
            bool otherGoingToStorage;
            {
                std::scoped_lock lock(otherState.mutex);
                otherPosX = otherState.posX;
                otherPosY = otherState.posY;
                otherVelX = otherState.velocityX;
                otherVelY = otherState.velocityY;
                otherGoingToStorage = otherState.goingToStorage;
            }
            
   
            bool otherIsEggTruck = (otherTruckId == 6);
            int otherStorageX = otherIsEggTruck ? storageEggX : storageFlourX;
            int otherStorageY = otherIsEggTruck ? storageEggY : storageFlourY;
            bool otherAtStorageUnloading = otherGoingToStorage && 
                                            std::abs(otherPosX - otherStorageX) < 5.0f &&
                                            std::abs(otherPosY - otherStorageY) < 5.0f;
            
            const int FRAMES_AHEAD = 17;
            
       
            if (!atStorageUnloading && !otherAtStorageUnloading) {
                
               
                bool willCollide = false;
                int collisionFrame = -1;
                float collisionPointX = 0.0f;
                float collisionPointY = 0.0f;
                
                for (int frame = 1; frame <= FRAMES_AHEAD; frame++) {
                  
                    float myPredictedX = truckFloatX + (moveX * frame);
                    float myPredictedY = truckFloatY + (moveY * frame);
                    
                    float otherPredictedX = otherPosX + (otherVelX * frame);
                    float otherPredictedY = otherPosY + (otherVelY * frame);
                    
                 
                    cugl::Rect myPredictedRect = cugl::Rect(
                        myPredictedX - truck->width / 2.0f,
                        myPredictedY - truck->height / 2.0f,
                        truck->width,
                        truck->height
                    );
                    
                    cugl::Rect otherPredictedRect = cugl::Rect(
                        otherPredictedX - truck->width / 2.0f,
                        otherPredictedY - truck->height / 2.0f,
                        truck->width,
                        truck->height
                    );
                    
                    if (myPredictedRect.doesIntersect(otherPredictedRect)) {
                        willCollide = true;
                        collisionFrame = frame;
                       
                        collisionPointX = (myPredictedX + otherPredictedX) / 2.0f;
                        collisionPointY = (myPredictedY + otherPredictedY) / 2.0f;
                        break; 
                    }
                }
                
                
                float currentDistX = truckFloatX - otherPosX;
                float currentDistY = truckFloatY - otherPosY;
                float currentDist = std::sqrt(currentDistX*currentDistX + currentDistY*currentDistY);
                
              
                {
                    std::scoped_lock lock(predictedCollision.mutex);
                    
                    if (willCollide) {
                      
                        if (predictedCollision.truckToStop == -1) {
                           
                            int initialChoice = (std::rand() % 2 == 0) ? 6 : 7;
                            
                            
                            bool chosenIsEggTruck = (initialChoice == 6);
                            bool otherIsEggTruck = !chosenIsEggTruck;
                            
                          
                            float chosenX, chosenY;
                            if (initialChoice == truckId) {
                                chosenX = truckFloatX;
                                chosenY = truckFloatY;
                            } else {
                                chosenX = otherPosX;
                                chosenY = otherPosY;
                            }
                            
                          
                            float otherStartX = otherIsEggTruck ? barn1X : barn2X;
                            float otherStartY = otherIsEggTruck ? barn1Y : barn2Y;
                            float otherEndX = otherIsEggTruck ? storageEggX : storageFlourX;
                            float otherEndY = otherIsEggTruck ? storageEggY : storageFlourY;
                            
                           
                            float pathDx = otherEndX - otherStartX;
                            float pathDy = otherEndY - otherStartY;
                            float pathLength = std::sqrt(pathDx * pathDx + pathDy * pathDy);
                            
                            bool chosenBlocksOther = false;
                            if (pathLength > 0) {
                                
                                float distance = std::abs(pathDy * (chosenX - otherStartX) - pathDx * (chosenY - otherStartY)) / pathLength;
                             
                                chosenBlocksOther = (distance < truck->width);
                            }
                            
           
                            if (chosenBlocksOther) {
                      
                                float otherX, otherY;
                                if (initialChoice == truckId) {
                                    otherX = otherPosX;
                                    otherY = otherPosY;
                                } else {
                                    otherX = truckFloatX;
                                    otherY = truckFloatY;
                                }
                                
                            
                                float chosenStartX = chosenIsEggTruck ? barn1X : barn2X;
                                float chosenStartY = chosenIsEggTruck ? barn1Y : barn2Y;
                                float chosenEndX = chosenIsEggTruck ? storageEggX : storageFlourX;
                                float chosenEndY = chosenIsEggTruck ? storageEggY : storageFlourY;
                                
                                float chosenPathDx = chosenEndX - chosenStartX;
                                float chosenPathDy = chosenEndY - chosenStartY;
                                float chosenPathLength = std::sqrt(chosenPathDx * chosenPathDx + chosenPathDy * chosenPathDy);
                                
                                bool otherBlocksChosen = false;
                                if (chosenPathLength > 0) {
                                    float distance = std::abs(chosenPathDy * (otherX - chosenStartX) - chosenPathDx * (otherY - chosenStartY)) / chosenPathLength;
                                    otherBlocksChosen = (distance < truck->width);
                                }
                                
                                
                                if (!otherBlocksChosen) {
                                    initialChoice = (initialChoice == 6) ? 7 : 6;
                                }
                             
                            }
                            
                            predictedCollision.collisionPredicted = true;
                            predictedCollision.truckToStop = initialChoice;
                        }
                        
                 
                        if (predictedCollision.truckToStop == truckId && !stoppedForTruck) {
                            stoppedForTruck = true;
                        }
                    } else {
                    
                        if (predictedCollision.collisionPredicted) {
                            predictedCollision.collisionPredicted = false;
                            predictedCollision.truckToStop = -1;
                        }
                        
                        if (stoppedForTruck) {
                            stoppedForTruck = false;
                        }
                    }
                }
            } else {
                
                if (stoppedForTruck) {
                    stoppedForTruck = false;
                }
             
                {
                    std::scoped_lock lock(predictedCollision.mutex);
                    if (predictedCollision.truckToStop == truckId) {
                        predictedCollision.collisionPredicted = false;
                        predictedCollision.truckToStop = -1;
                    }
                }
            }
        }
        

        if (!stoppedForTruck && (moveX != 0.0f || moveY != 0.0f)) {
           
            float dirX = targetX - truckFloatX;
            float dirY = targetY - truckFloatY;
            
           
            float actualMoveX = moveX;
            float actualMoveY = moveY;
            
            if (std::abs(moveX) > std::abs(dirX)) actualMoveX = dirX;
            if (std::abs(moveY) > std::abs(dirY)) actualMoveY = dirY;
            
            truckFloatX += actualMoveX;
            truckFloatY += actualMoveY;
        }
        
        
        int proposedX = static_cast<int>(std::round(truckFloatX));
        int proposedY = static_cast<int>(std::round(truckFloatY));
        
        bool shouldMove = (proposedX != truck->x || proposedY != truck->y);
        collisionMonitor->proposeMove(truckId, truck, proposedX, proposedY, shouldMove);
        barrier->proposalSubmitted();
        
   
        barrier->waitForCollisionResults(truckId);
        
     
        int blockedBy = -1, blockerX = 0, blockerY = 0;
        if (shouldMove) {
            if (!collisionMonitor->wasBlocked(truckId, blockedBy, blockerX, blockerY)) {
             
                truck->setPos(proposedX, proposedY);
                
                
                
                if (moveX < 0) {
                    truck->setDirection(1);   
                } else if (moveX > 0) {
                    truck->setDirection(-1);  
                }
                
                {
                    std::scoped_lock lock(farmMutex);
                    truck->updateFarm();
                }
            } else {
               
                truckFloatX = static_cast<float>(truck->x);
                truckFloatY = static_cast<float>(truck->y);
                {
                    std::scoped_lock lock(farmMutex);
                    truck->updateFarm();
                }
            }
        }
        
      
        barrier->threadFinished(truckId);
    }
}

void ovenThread(BakeryStorageMonitor* bakeryStorage,
                BakeryShopMonitor* shop,
                DisplayObject* bakeryCakes[],
                UpdateBarrier* barrier,
                Layer2CollisionMonitor* collisionMonitor) {
    bool baking = false;
    int bakingFrames = 0;
    bool ingredientsTaken = false;
    const int ovenId = 14;  
    
    while (running) {
        
        collisionMonitor->proposeMove(ovenId, nullptr, 0, 0, false);
        barrier->proposalSubmitted();
        
       
        barrier->waitForCollisionResults(ovenId);
        
      
        bakeryStorage->updateAnimation();
        
        if (baking) {
           
            bakingFrames++;
            
            if (bakingFrames >= 8) {
            shop->addCakes(3);
            
            {
                std::scoped_lock lock(farmMutex);
                    int cakeCount = shop->getCakeCount();
                    for (int i = 0; i < 6; i++) {
                        if (i < cakeCount) {
                    bakeryCakes[i]->updateFarm();
                        } else {
                            bakeryCakes[i]->erase();
                        }
                    }
                }
                
                baking = false;
                ingredientsTaken = false;  
            }
        } else if (bakeryStorage->isAnimating()) {
            
            
        } else if (ingredientsTaken) {
            
            baking = true;
            bakingFrames = 0;
            ingredientsTaken = false;
        } else {
            
            if (shop->getCakeCount() + 3 <= 6) {
                if (bakeryStorage->startTakeIngredients()) {
                    ingredientsTaken = true;
                }
            }
        }
        
        barrier->threadFinished(ovenId);
    }
}

void childThread(DisplayObject* child, int childId,
                 BakeryShopMonitor* shop,
                 SharedQueueState* queueState,
                 Layer2CollisionMonitor* collisionMonitor,
                 std::vector<DisplayObject*>& allLayer2,
                 UpdateBarrier* barrier) {
    
    const int FIXED_POSITIONS[5][2] = {
        {760, 200},  
        {760, 250},  
        {760, 300},  
        {760, 350},  
        {760, 400}   
    };
    
    enum ChildState {
        MOVING_TO_POSITION,  
        AT_POSITION          
    };
    
    
    enum ReturnWaypoint {
        RETURN_DIRECT,      
        RETURN_GO_LEFT,     
        RETURN_GO_UP,       
        RETURN_GO_RIGHT     
    };
    
    ChildState state = MOVING_TO_POSITION;
    ReturnWaypoint returnWaypoint = RETURN_DIRECT;
    int cakesWanted = 0;
    const int detourX = 690; 
    
    while (running) {
     
        int myPosition = queueState->getPosition(childId);
        
       
        int targetX = FIXED_POSITIONS[myPosition][0];
        int targetY = FIXED_POSITIONS[myPosition][1];
        
            
            int dx = 0, dy = 0;
        int proposedX = child->x;
        int proposedY = child->y;
        bool shouldMove = false;
        
     
        int speed = 2;  
        
        if (myPosition == 4 && returnWaypoint != RETURN_DIRECT) {
           
            speed = 10;  
        }
        
        
        if (returnWaypoint == RETURN_DIRECT) {
            
            if (child->x < targetX) dx = std::min(speed, targetX - child->x);
            else if (child->x > targetX) dx = -std::min(speed, child->x - targetX);
            
            if (child->y < targetY) dy = std::min(speed, targetY - child->y);
            else if (child->y > targetY) dy = -std::min(speed, child->y - targetY);
            
        } else if (returnWaypoint == RETURN_GO_LEFT) {
            
            if (child->x > detourX) {
                dx = -std::min(speed, child->x - detourX);
            } else {
                returnWaypoint = RETURN_GO_UP;
            }
            
        } else if (returnWaypoint == RETURN_GO_UP) {
          
            if (child->y < targetY) {
                dy = std::min(speed, targetY - child->y);
            } else {
                returnWaypoint = RETURN_GO_RIGHT;
            }
            
        } else if (returnWaypoint == RETURN_GO_RIGHT) {
         
            if (child->x < targetX) dx = std::min(speed, targetX - child->x);
            else if (child->x > targetX) dx = -std::min(speed, child->x - targetX);
            
            if (child->y < targetY) dy = std::min(speed, targetY - child->y);
            else if (child->y > targetY) dy = -std::min(speed, child->y - targetY);
            
          
            if (dx == 0 && dy == 0) {
                returnWaypoint = RETURN_DIRECT;
            }
        }
        
        proposedX = child->x + dx;
        proposedY = child->y + dy;
        shouldMove = (dx != 0 || dy != 0);
        
        collisionMonitor->proposeMove(childId, child, proposedX, proposedY, shouldMove);
        barrier->proposalSubmitted();
        
        
        barrier->waitForCollisionResults(childId);
        
        
        int blockedBy = -1, blockerX = 0, blockerY = 0;
        if (shouldMove) {
            if (!collisionMonitor->wasBlocked(childId, blockedBy, blockerX, blockerY)) {
                
                child->setPos(proposedX, proposedY);
                {
                    
                    std::scoped_lock lock(farmMutex);
                    child->updateFarm();
                    
                }
            } else {
                
                std::scoped_lock lock(farmMutex);
                child->updateFarm();
            }
            } else {
            
                    std::scoped_lock lock(farmMutex);
                    child->updateFarm();
                }
        
        
        if (child->x == targetX && child->y == targetY) {
           
            state = AT_POSITION;
            
           
            if (myPosition == 0) {
               
                if (cakesWanted == 0) {
                   
                    bool entered = shop->enterShop();
                    if (entered) {
                        cakesWanted = 1 + (std::rand() % 6);
                        
                    }
                }
                
              
                if (cakesWanted > 0) {
                int bought = shop->buyCakes(cakesWanted);
                cakesWanted -= bought;
                
            
                  
                    if (cakesWanted == 0) {
                        
            shop->leaveShop();
                        queueState->childFinishedShopping(childId);
                        
                        returnWaypoint = RETURN_GO_LEFT;
                        state = MOVING_TO_POSITION;  
                    }
                }
            }
        } else {
            
            state = MOVING_TO_POSITION;
        }
        
        
        barrier->threadFinished(childId);
    }
}

void cowThread(DisplayObject* cow, int cowId, Layer2CollisionMonitor* collisionMonitor, std::vector<DisplayObject*>& allLayer2, UpdateBarrier* barrier) {
    
    
    while (running) {
        
        int proposedX = cow->x;
        int proposedY = cow->y;
        bool shouldMove = false;  
        
        collisionMonitor->proposeMove(cowId, cow, proposedX, proposedY, shouldMove);
        barrier->proposalSubmitted();
        
       
        barrier->waitForCollisionResults(cowId);
        
    
            {
                std::scoped_lock lock(farmMutex);
                cow->updateFarm();
        }
        
        
        barrier->threadFinished(cowId);
    }
}



void FarmLogic::run() {
    BakeryStats stats;
    globalStats = &stats;
    std::srand(std::time(0));
    
    
    std::vector<NestMonitor*> nests;
    nests.push_back(new NestMonitor(100, 490));
    nests.push_back(new NestMonitor(400, 490));
    nests.push_back(new NestMonitor(700, 490));
    
    BarnMonitor* barn1 = new BarnMonitor(true);   
    BarnMonitor* barn2 = new BarnMonitor(false);  
    
    SharedQueueState* childQueue = new SharedQueueState();
    
    UpdateBarrier* barrier = new UpdateBarrier(15);  
    
    
    Layer2CollisionMonitor* collisionMonitor = new Layer2CollisionMonitor(15);  
    
  
    DisplayObject chicken1("chicken", 24, 24, 2, 0);
    DisplayObject chicken2("chicken", 24, 24, 2, 1);
    DisplayObject chicken3("chicken", 24, 24, 2, 2);
    DisplayObject chicken4("chicken", 24, 24, 2, 3);
    
   
    std::vector<DisplayObject*> allLayer2Objects;
    
    DisplayObject nestObj1("nest", 80, 60, 0, 14);
    DisplayObject nestObj2("nest", 80, 60, 0, 15);
    DisplayObject nestObj3("nest", 80, 60, 0, 16);
    
   
    DisplayObject* nestObjects[3] = {
        &nestObj1, &nestObj2, &nestObj3
    };
    
    DisplayObject cow1("cow", 40, 40, 2, 4);
    DisplayObject cow2("cow", 40, 40, 2, 5);
    
    DisplayObject truck1("truck", 35, 26, 2, 6);
    DisplayObject truck2("truck", 35, 26, 2, 7); 
    
    DisplayObject farmer("farmer", 25, 40, 2, 8);
    
    DisplayObject child1("child", 25, 40, 2, 9);
    DisplayObject child2("child", 25, 40, 2, 10);
    DisplayObject child3("child", 25, 40, 2, 11);
    DisplayObject child4("child", 25, 40, 2, 12);
    DisplayObject child5("child", 25, 40, 2, 13);
    
   
    allLayer2Objects.push_back(&chicken1);
    allLayer2Objects.push_back(&chicken2);
    allLayer2Objects.push_back(&chicken3);
    allLayer2Objects.push_back(&chicken4);
    allLayer2Objects.push_back(&cow1);
    allLayer2Objects.push_back(&cow2);
    allLayer2Objects.push_back(&truck1);
    allLayer2Objects.push_back(&truck2);
    allLayer2Objects.push_back(&farmer);
    allLayer2Objects.push_back(&child1);
    allLayer2Objects.push_back(&child2);
    allLayer2Objects.push_back(&child3);
    allLayer2Objects.push_back(&child4);
    allLayer2Objects.push_back(&child5);
    
    DisplayObject barn1Obj("barn", 100, 100, 0, 17);
    DisplayObject barn2Obj("barn", 100, 100, 0, 18);
    DisplayObject bakery("bakery", 250, 250, 0, 19);
    
 
    DisplayObject nestEggObj0("egg", 10, 20, 1, 20);
    DisplayObject nestEggObj1("egg", 10, 20, 1, 21);
    DisplayObject nestEggObj2("egg", 10, 20, 1, 22);
    DisplayObject nestEggObj3("egg", 10, 20, 1, 23);
    DisplayObject nestEggObj4("egg", 10, 20, 1, 24);
    DisplayObject nestEggObj5("egg", 10, 20, 1, 25);
    DisplayObject nestEggObj6("egg", 10, 20, 1, 26);
    DisplayObject nestEggObj7("egg", 10, 20, 1, 27);
    DisplayObject nestEggObj8("egg", 10, 20, 1, 28);
    

    DisplayObject* nestEggs[9] = {
        &nestEggObj0, &nestEggObj1, &nestEggObj2,
        &nestEggObj3, &nestEggObj4, &nestEggObj5,
        &nestEggObj6, &nestEggObj7, &nestEggObj8
    };
    
    DisplayObject cakeObj0("cake", 20, 20, 1, 29);
    DisplayObject cakeObj1("cake", 20, 20, 1, 30);
    DisplayObject cakeObj2("cake", 20, 20, 1, 31);
    DisplayObject cakeObj3("cake", 20, 20, 1, 32);
    DisplayObject cakeObj4("cake", 20, 20, 1, 33);
    DisplayObject cakeObj5("cake", 20, 20, 1, 34);
    
    DisplayObject* bakerycake[6] = {
        &cakeObj0, &cakeObj1, &cakeObj2,
        &cakeObj3, &cakeObj4, &cakeObj5
    };
    
   
    DisplayObject bakeryEggObj0("egg", 15, 15, 1, 35);
    DisplayObject bakeryEggObj1("egg", 15, 15, 1, 36);
    DisplayObject bakeryEggObj2("egg", 15, 15, 1, 37);
    DisplayObject bakeryEggObj3("egg", 15, 15, 1, 38);
    DisplayObject bakeryEggObj4("egg", 15, 15, 1, 39);
    DisplayObject bakeryEggObj5("egg", 15, 15, 1, 40);
    
    DisplayObject bakeryButterObj0("butter", 15, 15, 1, 41);
    DisplayObject bakeryButterObj1("butter", 15, 15, 1, 42);
    DisplayObject bakeryButterObj2("butter", 15, 15, 1, 43);
    DisplayObject bakeryButterObj3("butter", 15, 15, 1, 44);
    DisplayObject bakeryButterObj4("butter", 15, 15, 1, 45);
    DisplayObject bakeryButterObj5("butter", 15, 15, 1, 46);
    
    DisplayObject bakeryFlourObj0("flour", 15, 15, 1, 47);
    DisplayObject bakeryFlourObj1("flour", 15, 15, 1, 48);
    DisplayObject bakeryFlourObj2("flour", 15, 15, 1, 49);
    DisplayObject bakeryFlourObj3("flour", 15, 15, 1, 50);
    DisplayObject bakeryFlourObj4("flour", 15, 15, 1, 51);
    DisplayObject bakeryFlourObj5("flour", 15, 15, 1, 52);
    
    DisplayObject bakerySugarObj0("sugar", 15, 15, 1, 53);
    DisplayObject bakerySugarObj1("sugar", 15, 15, 1, 54);
    DisplayObject bakerySugarObj2("sugar", 15, 15, 1, 55);
    DisplayObject bakerySugarObj3("sugar", 15, 15, 1, 56);
    DisplayObject bakerySugarObj4("sugar", 15, 15, 1, 57);
    DisplayObject bakerySugarObj5("sugar", 15, 15, 1, 58);
    
    
    DisplayObject travelEggObj0("egg", 15, 15, 1, 59);
    DisplayObject travelEggObj1("egg", 15, 15, 1, 60);
    DisplayObject travelButterObj0("butter", 15, 15, 1, 61);
    DisplayObject travelButterObj1("butter", 15, 15, 1, 62);
    DisplayObject travelFlourObj0("flour", 15, 15, 1, 63);
    DisplayObject travelFlourObj1("flour", 15, 15, 1, 64);
    DisplayObject travelSugarObj0("sugar", 15, 15, 1, 65);
    DisplayObject travelSugarObj1("sugar", 15, 15, 1, 66);
    
    DisplayObject* bakeryEggs[6] = {
        &bakeryEggObj0, &bakeryEggObj1, &bakeryEggObj2,
        &bakeryEggObj3, &bakeryEggObj4, &bakeryEggObj5
    };
    
    DisplayObject* bakeryButter[6] = {
        &bakeryButterObj0, &bakeryButterObj1, &bakeryButterObj2,
        &bakeryButterObj3, &bakeryButterObj4, &bakeryButterObj5
    };
    
    DisplayObject* bakeryFlour[6] = {
        &bakeryFlourObj0, &bakeryFlourObj1, &bakeryFlourObj2,
        &bakeryFlourObj3, &bakeryFlourObj4, &bakeryFlourObj5
    };
    
    DisplayObject* bakerySugar[6] = {
        &bakerySugarObj0, &bakerySugarObj1, &bakerySugarObj2,
        &bakerySugarObj3, &bakerySugarObj4, &bakerySugarObj5
    };
    
    DisplayObject* travelEggs[2] = {
        &travelEggObj0, &travelEggObj1
    };
    
    DisplayObject* travelButter[2] = {
        &travelButterObj0, &travelButterObj1
    };
    
    DisplayObject* travelFlour[2] = {
        &travelFlourObj0, &travelFlourObj1
    };
    
    DisplayObject* travelSugar[2] = {
        &travelSugarObj0, &travelSugarObj1
    };

    chicken1.setPos(150, 450);
    chicken1.setDirection(1);  
    chicken2.setPos(400, 470);
    chicken2.setDirection(1);  
    chicken3.setPos(650, 420);
    chicken3.setDirection(1);  
    chicken4.setPos(300, 500);
    chicken4.setDirection(1);  
    
    nestObj1.setPos(nests[0]->getX(), nests[0]->getY());
    nestObj2.setPos(nests[1]->getX(), nests[1]->getY());
    nestObj3.setPos(nests[2]->getX(), nests[2]->getY());
    
    for (int i = 0; i < 3; i++) {
        nestEggs[i * 3 + 0]->setPos(nests[i]->getX() - 10, nests[i]->getY() + 7);
        nestEggs[i * 3 + 1]->setPos(nests[i]->getX(), nests[i]->getY() + 7);
        nestEggs[i * 3 + 2]->setPos(nests[i]->getX() + 10, nests[i]->getY() + 7);
    }
    
    
    cow1.setPos(250, 350);  
    cow2.setPos(450, 350);  
    
    
    truck1.setPos(90, 50);
    truck1.setDirection(-1);  
    truck2.setPos(90, 150);
    truck2.setDirection(-1);  
    
    farmer.setPos(100, 530);  
    
    
    
    child1.setPos(760, 200);  
    child2.setPos(760, 250);  
    child3.setPos(760, 300);  
    child4.setPos(760, 350);  
    child5.setPos(760, 400);  
    
    barn1Obj.setPos(50, 50);
    barn2Obj.setPos(50, 150);
    bakery.setPos(650, 150);
    
  
    bakerycake[0]->setPos(680, 90);   
    bakerycake[1]->setPos(700, 90);
    bakerycake[2]->setPos(720, 90);
    bakerycake[3]->setPos(680, 110);  
    bakerycake[4]->setPos(700, 110);
    bakerycake[5]->setPos(720, 110);
    
   
    for (int i = 0; i < 6; i++) {
        bakerycake[i]->erase();
    }
    
    
    BakeryShopMonitor* bakeryShop = new BakeryShopMonitor(bakerycake);
    
    
    int stockBaseX = 530;  
    int stockBaseY = 160;  
    int spacing = 20;      
    
    
    for (int i = 0; i < 6; i++) {
        bakeryEggs[i]->setPos(stockBaseX + i * spacing, stockBaseY);
        bakeryEggs[i]->erase();  
    }
   
    for (int i = 0; i < 6; i++) {
        bakeryButter[i]->setPos(stockBaseX + i * spacing, stockBaseY + spacing);
        bakeryButter[i]->erase();  
    }
    
    for (int i = 0; i < 6; i++) {
        bakeryFlour[i]->setPos(stockBaseX + i * spacing, stockBaseY + spacing * 2);
        bakeryFlour[i]->erase();  
    }
   
    for (int i = 0; i < 6; i++) {
        bakerySugar[i]->setPos(stockBaseX + i * spacing, stockBaseY + spacing * 3);
        bakerySugar[i]->erase(); 
    }
    
   
    BakeryStorageMonitor* bakeryStorage = new BakeryStorageMonitor(
        bakeryEggs, bakeryButter, bakeryFlour, bakerySugar,
        travelEggs, travelButter, travelFlour, travelSugar,
        stockBaseX, stockBaseY, spacing
    );
    
  
    chicken1.updateFarm();
    chicken2.updateFarm();
    chicken3.updateFarm();
    chicken4.updateFarm();
    nestObj1.updateFarm();
    nestObj2.updateFarm();
    nestObj3.updateFarm();
    cow1.updateFarm();
    cow2.updateFarm();
    truck1.updateFarm();
    truck2.updateFarm();
    farmer.updateFarm();
    child1.updateFarm();
    child2.updateFarm();
    child3.updateFarm();
    child4.updateFarm();
    child5.updateFarm();
    
    barn1Obj.updateFarm();
    barn2Obj.updateFarm();
    bakery.updateFarm();
    
    DisplayObject::redisplay(stats);
    

    std::vector<std::thread> threads;
    
    threads.emplace_back(redisplayThread, &stats, barrier, collisionMonitor, nestObjects, bakeryShop, barn1);
    
    threads.emplace_back(chickenThread, &chicken1, 0, std::ref(nests), nestObjects, nestEggs, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(chickenThread, &chicken2, 1, std::ref(nests), nestObjects, nestEggs, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(chickenThread, &chicken3, 2, std::ref(nests), nestObjects, nestEggs, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(chickenThread, &chicken4, 3, std::ref(nests), nestObjects, nestEggs, collisionMonitor, std::ref(allLayer2Objects), barrier);
    
    threads.emplace_back(cowThread, &cow1, 4, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(cowThread, &cow2, 5, collisionMonitor, std::ref(allLayer2Objects), barrier);
    
    threads.emplace_back(truckThread, &truck1, 6, barn1, bakeryStorage, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(truckThread, &truck2, 7, barn2, bakeryStorage, collisionMonitor, std::ref(allLayer2Objects), barrier);
    
    threads.emplace_back(farmerThread, &farmer, 8, std::ref(nests), nestEggs, barn1, collisionMonitor, std::ref(allLayer2Objects), barrier);
    
    threads.emplace_back(ovenThread, bakeryStorage, bakeryShop, bakerycake, barrier, collisionMonitor);
    
    threads.emplace_back(childThread, &child1, 9, bakeryShop, childQueue, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(childThread, &child2, 10, bakeryShop, childQueue, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(childThread, &child3, 11, bakeryShop, childQueue, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(childThread, &child4, 12, bakeryShop, childQueue, collisionMonitor, std::ref(allLayer2Objects), barrier);
    threads.emplace_back(childThread, &child5, 13, bakeryShop, childQueue, collisionMonitor, std::ref(allLayer2Objects), barrier);
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    
    for (auto nest : nests) delete nest;
    delete barn1;
    delete barn2;
    delete bakeryStorage;
    delete bakeryShop;
    delete collisionMonitor;
    delete barrier;
}

void FarmLogic::start() {
    std::thread([]() {
       FarmLogic::run();
    })
    .detach();
}
