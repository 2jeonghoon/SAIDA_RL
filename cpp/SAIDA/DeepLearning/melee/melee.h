/*
 * SCV-based Resource Gathering and Building Environment
 * Based on the structure from the SAIDA team's examples.
 */

#pragma once
#include "../Gym.h"
#include "../RLSharedMemory.h"

using namespace Message;

namespace BWML {

    class melee : public Gym
    {
    private:
        // --- Agent and Game Objects ---
        Unit agent = nullptr;           // Our SCV
        Unit commandCenter = nullptr;   // The main command center
        Unit supplyDepot = nullptr;     // The supply depot we are building
        vector<Unit> mineralPatches;    // List of nearby mineral patches

        // --- Action Space Definition ---
        const int GATHER_ACTION_COUNT = 8;
        const int ACTION_RETURN_CARGO = 8;
        const int ACTION_BUILD_SUPPLY_DEPOT = 9;
        const int ACTION_SIZE = 10;

        // --- Episode State ---
        Position buildLocation = Positions::None;
        int lastAction = -1;
        bool invalidAction = false;
        int lastMineralCount = 50; // Initial minerals

    protected:
        // --- Gym Override Functions ---
        void init(::google::protobuf::Message* message) override;
        bool isDone() override;
        void reset(bool isFirstResetCall) override;
        float getReward() override;
        bool isResetFinished() override;
        bool isActionFinished() override;
        void makeInitMessage(::google::protobuf::Message* message) override;
        void getObservation(::google::protobuf::Message* stateMsg) override;
        bool initializeAndValidate() override;

    public:
        melee(string shmName, ConnMethod method = SHARED_MEMORY) : Gym("melee") {
            connection = new RLSharedMemory(shmName, 2000000);
            connection->initialize();
        }
        ~melee() { };

        // Singleton Instance getter
        static melee& Instance(string shmName = "");

        void step(::google::protobuf::Message* stepReqMsg) override;
        void render() override;
        void onUnitDestroy(Unit unit) override;
    };
}