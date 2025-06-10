w/*
 * SCV-based Resource Gathering and Building Environment
 * Based on the structure from the SAIDA team's examples.
 */

#include "melee.h"
#include "../../UXManager.h"

using namespace BWML;
using namespace MyBot;

// Singleton Instance
melee& melee::Instance(string shmName) {
    static melee instance(shmName);
    return instance;
}

// Init: Called once at the beginning from the Python side.
void melee::init(::google::protobuf::Message* message)
{
    InitReq* initReq = (InitReq*)message;
    STEP_FRAME = initReq->frames_per_step();
    // In this environment, action space is fixed, so we don't need much from initReq.
}

bool melee::initializeAndValidate()
{
    // All validation happens in reset() for this environment.
    return true;
}

// makeInitMessage: Sends environment info (action size, unit types) to Python.
void melee::makeInitMessage(::google::protobuf::Message* initMessage)
{
    InitRes* message = (InitRes*)initMessage;

    message->set_num_action_space(ACTION_SIZE);

    // Define the unit types the agent will interact with.
    setTypeInfo(message->add_unit_type_map(), Terran_SCV);
    setTypeInfo(message->add_unit_type_map(), Terran_Command_Center);
    setTypeInfo(message->add_unit_type_map(), Terran_Supply_Depot);
    setTypeInfo(message->add_unit_type_map(), Resource_Mineral_Field);
}

// Reset: Called at the start of each new episode.
void melee::reset(bool isFirstResetCall)
{
    // On repeated resets, restart the game to ensure a clean state
    if (agent && agent->getID() > 9000) {
        restartGame();
    }

    agent = nullptr;
    commandCenter = nullptr;
    supplyDepot = nullptr;
    mineralPatches.clear();

    buildLocation = Positions::None;
    lastAction = -1;
    invalidAction = false;
    lastMineralCount = 50;
}

// isResetFinished: Checks if the initial conditions for an episode are met.
bool melee::isResetFinished()
{
    printf("melee\n");
    // 1. Find the SCV
    if (agent == nullptr && INFO.getUnits(Terran_SCV, S).size() > 0) {
        agent = INFO.getUnits(Terran_SCV, S)[0]->unit();
    }

    // 2. Find the Command Center
    if (commandCenter == nullptr && INFO.getUnits(Terran_Command_Center, S).size() > 0) {
        commandCenter = INFO.getUnits(Terran_Command_Center, S)[0]->unit();
    }

    // 3. Once both are found, set up the environment
    if (agent && commandCenter) {
        // Find and sort mineral patches by distance to the command center
        if (mineralPatches.empty()) {
            uList allMinerals = INFO.getUnits(Resource_Mineral_Field, AllPlayers);
            for (auto m : allMinerals) {
                mineralPatches.push_back(m->unit());
            }
            sort(mineralPatches.begin(), mineralPatches.end(), [&](Unit a, Unit b) {
                return a->getDistance(commandCenter) < b->getDistance(commandCenter);
            });
            // Keep only the closest ones needed for our action space
            if (mineralPatches.size() > GATHER_ACTION_COUNT) {
                mineralPatches.resize(GATHER_ACTION_COUNT);
            }
        }

        // Define a valid build location (e.g., to the right of the Command Center)
        if (buildLocation == Positions::None) {
            buildLocation = TilePosition(commandCenter->getTilePosition().x + 5, commandCenter->getTilePosition().y);
        }

        // Check if everything is ready
        if (!mineralPatches.empty() && buildLocation.isValid()) {
            return true;
        }
    }

    return false;
}

// Step: Executes an action received from the agent.
void melee::step(::google::protobuf::Message* stepReqMsg)
{
    if (!agent || !agent->exists()) return;

    // Reset flags
    invalidAction = false;
    Action act = ((StepReq*)stepReqMsg)->action(0);
    lastAction = act.action_num();

    // --- Action Logic ---
    if (lastAction < GATHER_ACTION_COUNT) { // Action 0-7: Gather
        if (lastAction < mineralPatches.size()) {
            Unit targetMineral = mineralPatches[lastAction];
            if (targetMineral && targetMineral->exists()) {
                agent->gather(targetMineral);
            } else {
                invalidAction = true;
            }
        } else {
            invalidAction = true;
        }
    }
    else if (lastAction == ACTION_RETURN_CARGO) { // Action 8: Return Cargo
        agent->returnCargo();
    }
    else if (lastAction == ACTION_BUILD_SUPPLY_DEPOT) { // Action 9: Build
        if (Broodwar->self()->minerals() >= 100) {
            agent->build(Terran_Supply_Depot, buildLocation);
        } else {
            invalidAction = true;
        }
    }
}

// isActionFinished: Determines when the current step/action is complete.
bool melee::isActionFinished()
{
    if (isDone()) {
        return true;
    }

    // A simple frame-based step completion is reliable.
    if (STEP_FRAME != -1) {
        return (TIME - startFrame) % STEP_FRAME == 0;
    }

    // Fallback: wait 8 frames (a common default)
    return (TIME - startFrame) % 8 == 0;
}

// getObservation: Gathers game state and sends it to the agent.
void melee::getObservation(::google::protobuf::Message* stateMsg)
{
    State* stateMessage = (State*)stateMsg;
    if (!agent || !agent->exists()) return;

    // My SCV's info
    setUInfo(stateMessage->add_my_unit(), agent);

    // Enemy units list will be used for other game objects
    // Command Center info
    if (commandCenter && commandCenter->exists()) {
        setUInfo(stateMessage->add_en_unit(), INFO.getUnitInfo(commandCenter, S));
    }
    
    // Mineral patches info
    for(const auto& mineral : mineralPatches) {
        if (mineral && mineral->exists()) {
            setUInfo(stateMessage->add_en_unit(), INFO.getUnitInfo(mineral, AllPlayers));
        }
    }

    // Supply Depot info (if it exists)
    uList supplyDepots = INFO.getUnits(Terran_Supply_Depot, S);
    if (!supplyDepots.empty()) {
        // Find the one closest to our build location
        UnitInfo* depotInfo = nullptr;
        int min_dist = INT_MAX;

        for (auto si : supplyDepots) {
            int dist = si->pos().getApproxDistance(Position(buildLocation));
            if (dist < min_dist) {
                min_dist = dist;
                depotInfo = si;
            }
        }
        if (depotInfo) {
            setUInfo(stateMessage->add_en_unit(), depotInfo);
            supplyDepot = depotInfo->unit(); // Keep a reference to it
        }
    }
}

// isDone: Checks for episode termination conditions.
bool melee::isDone()
{
    // Success: Supply Depot is completed.
    if (supplyDepot && supplyDepot->isCompleted()) {
        return true;
    }

    // Failure: SCV is destroyed.
    if (agent && !agent->exists()) {
        return true;
    }

    return false;
}

// getReward: Calculates and returns the reward for the last action.
float melee::getReward()
{
    // Final reward for ending the episode
    if (isDone()) {
        if (supplyDepot && supplyDepot->isCompleted()) {
            return 200.0f; // Success
        }
        return -200.0f; // Failure (SCV died)
    }

    // Step-by-step rewards
    float reward = -0.01f; // Small time penalty

    if (invalidAction) {
        reward -= 1.0f;
    }

    // Reward for delivering minerals
    int currentMinerals = Broodwar->self()->minerals();
    if (currentMinerals > lastMineralCount) {
        reward += 10.0f;
    }
    lastMineralCount = currentMinerals;
    
    // Reward for starting construction
    if (agent && agent->isConstructing() && lastAction == ACTION_BUILD_SUPPLY_DEPOT) {
         // Check if this is the first frame of construction
        Unit target = agent->getOrderTarget();
        if (target && target->getType() == Terran_Supply_Depot && !target->isCompleted()) {
             // A simple way to give this reward only once is to check build progress
            if (target->getHitPoints() < 50) { // Hasn't been building for long
                 reward += 50.0f;
            }
        }
    }

    return reward;
}

// render: Draws debug information on the screen.
void melee::render()
{
    if (!agent || !agent->exists()) return;

    // Focus camera on the agent
    focus(agent->getPosition());

    // Draw a circle around the SCV
    bw->drawCircleMap(agent->getPosition(), 20, Colors::Green);

    // Draw the target build location
    bw->drawBoxMap(Position(buildLocation), Position(buildLocation) + Position(96, 64), Colors::Cyan);
    
    // Draw lines to mineral patches
    for(const auto& mineral : mineralPatches) {
        if (mineral && mineral->exists()) {
            bw->drawLineMap(agent->getPosition(), mineral->getPosition(), Colors::Grey);
        }
    }

    // Display SCV status
    string status = "Idle";
    if (agent->isGatheringMinerals()) status = "Gathering";
    if (agent->isReturningCargo()) status = "Returning";
    if (agent->isConstructing()) status = "Building";
    bw->drawTextMap(agent->getPosition() + Position(0, 20), "Status: %s", status.c_str());
    bw->drawTextMap(agent->getPosition() + Position(0, 30), "Minerals: %d", Broodwar->self()->minerals());
}

// onUnitDestroy: Callback for when a unit is destroyed.
void melee::onUnitDestroy(Unit unit)
{
    if (agent && unit->getID() == agent->getID()) {
        agent = nullptr;
    }
}