#include "Solver.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

#include <cmath>

#include "MpSolver.h"
#include "CsvReader.h"


using namespace std;


namespace szx {

#pragma region Solver::Cli
int Solver::Cli::run(int argc, char * argv[]) {
    Log(LogSwitch::Szx::Cli) << "parse command line arguments." << endl;
    Set<String> switchSet;
    Map<String, char*> optionMap({ // use string as key to compare string contents instead of pointers.
        { InstancePathOption(), nullptr },
        { SolutionPathOption(), nullptr },
        { RandSeedOption(), nullptr },
        { TimeoutOption(), nullptr },
        { MaxIterOption(), nullptr },
        { JobNumOption(), nullptr },
        { RunIdOption(), nullptr },
        { EnvironmentPathOption(), nullptr },
        { ConfigPathOption(), nullptr },
        { LogPathOption(), nullptr }
    });

    for (int i = 1; i < argc; ++i) { // skip executable name.
        auto mapIter = optionMap.find(argv[i]);
        if (mapIter != optionMap.end()) { // option argument.
            mapIter->second = argv[++i];
        } else { // switch argument.
            switchSet.insert(argv[i]);
        }
    }

    Log(LogSwitch::Szx::Cli) << "execute commands." << endl;
    if (switchSet.find(HelpSwitch()) != switchSet.end()) {
        cout << HelpInfo() << endl;
    }

    if (switchSet.find(AuthorNameSwitch()) != switchSet.end()) {
        cout << AuthorName() << endl;
    }

    Solver::Environment env;
    env.load(optionMap);
    if (env.instPath.empty() || env.slnPath.empty()) { return -1; }

    Solver::Configuration cfg;
    cfg.load(env.cfgPath);

    Log(LogSwitch::Szx::Input) << "load instance " << env.instPath << " (seed=" << env.randSeed << ")." << endl;
    Problem::Input input;
    if (!input.load(env.instPath)) { return -1; }

    Solver solver(input, env, cfg);
    solver.solve();

    pb::Submission submission;
    submission.set_thread(to_string(env.jobNum));
    submission.set_instance(env.friendlyInstName());
    submission.set_duration(to_string(solver.timer.elapsedSeconds()) + "s");

    solver.output.save(env.slnPath, submission);
    #if SZX_DEBUG
    solver.output.save(env.solutionPathWithTime(), submission);
    solver.record();
    #endif // SZX_DEBUG

    return 0;
}
#pragma endregion Solver::Cli

#pragma region Solver::Environment
void Solver::Environment::load(const Map<String, char*> &optionMap) {
    char *str;

    str = optionMap.at(Cli::EnvironmentPathOption());
    if (str != nullptr) { loadWithoutCalibrate(str); }

    str = optionMap.at(Cli::InstancePathOption());
    if (str != nullptr) { instPath = str; }

    str = optionMap.at(Cli::SolutionPathOption());
    if (str != nullptr) { slnPath = str; }

    str = optionMap.at(Cli::RandSeedOption());
    if (str != nullptr) { randSeed = atoi(str); }

    str = optionMap.at(Cli::TimeoutOption());
    if (str != nullptr) { msTimeout = static_cast<Duration>(atof(str) * Timer::MillisecondsPerSecond); }

    str = optionMap.at(Cli::MaxIterOption());
    if (str != nullptr) { maxIter = atoi(str); }

    str = optionMap.at(Cli::JobNumOption());
    if (str != nullptr) { jobNum = atoi(str); }

    str = optionMap.at(Cli::RunIdOption());
    if (str != nullptr) { rid = str; }

    str = optionMap.at(Cli::ConfigPathOption());
    if (str != nullptr) { cfgPath = str; }

    str = optionMap.at(Cli::LogPathOption());
    if (str != nullptr) { logPath = str; }

    calibrate();
}

void Solver::Environment::load(const String &filePath) {
    loadWithoutCalibrate(filePath);
    calibrate();
}

void Solver::Environment::loadWithoutCalibrate(const String &filePath) {
    // EXTEND[szx][8]: load environment from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Environment::save(const String &filePath) const {
    // EXTEND[szx][8]: save environment to file.
}
void Solver::Environment::calibrate() {
    // adjust thread number.
    int threadNum = thread::hardware_concurrency();
    if ((jobNum <= 0) || (jobNum > threadNum)) { jobNum = threadNum; }

    // adjust timeout.
    msTimeout -= Environment::SaveSolutionTimeInMillisecond;
}
#pragma endregion Solver::Environment

#pragma region Solver::Configuration
void Solver::Configuration::load(const String &filePath) {
    // EXTEND[szx][5]: load configuration from file.
    // EXTEND[szx][8]: check file existence first.
}

void Solver::Configuration::save(const String &filePath) const {
    // EXTEND[szx][5]: save configuration to file.
}
#pragma endregion Solver::Configuration

#pragma region Solver
bool Solver::solve() {
    init();

    int workerNum = (max)(1, env.jobNum / cfg.threadNumPerWorker);
    cfg.threadNumPerWorker = env.jobNum / workerNum;
    List<Solution> solutions(workerNum, Solution(this));
    List<bool> success(workerNum);

    Log(LogSwitch::Szx::Framework) << "launch " << workerNum << " workers." << endl;
    List<thread> threadList;
    threadList.reserve(workerNum);
    for (int i = 0; i < workerNum; ++i) {
        // TODO[szx][2]: as *this is captured by ref, the solver should support concurrency itself, i.e., data members should be read-only or independent for each worker.
        // OPTIMIZE[szx][3]: add a list to specify a series of algorithm to be used by each threads in sequence.
        threadList.emplace_back([&, i]() { success[i] = optimize(solutions[i], i); });
    }
    for (int i = 0; i < workerNum; ++i) { threadList.at(i).join(); }

    Log(LogSwitch::Szx::Framework) << "collect best result among all workers." << endl;
    int bestIndex = -1;
    Length bestValue = Problem::MaxDistance;
    for (int i = 0; i < workerNum; ++i) {
        if (!success[i]) { continue; }
        Log(LogSwitch::Szx::Framework) << "worker " << i << " got " << solutions[i].coverRadius << endl;
        if (solutions[i].coverRadius >= bestValue) { continue; }
        bestIndex = i;
        bestValue = solutions[i].coverRadius;
    }

    env.rid = to_string(bestIndex);
    if (bestIndex < 0) { return false; }
    output = solutions[bestIndex];
    return true;
}

void Solver::record() const {
    #if SZX_DEBUG
    int generation = 0;

    ostringstream log;

    System::MemoryUsage mu = System::peakMemoryUsage();

    Length obj = output.coverRadius;
    Length checkerObj = -1;
    bool feasible = check(checkerObj);

    // record basic information.
    log << env.friendlyLocalTime() << ","
        << env.rid << ","
        << env.instPath << ","
        << feasible << "," << (obj - checkerObj) << ",";
    if (Problem::isTopologicalGraph(input)) {
        log << obj << ",";
    } else {
        auto oldPrecision = log.precision();
        log.precision(2);
        log << fixed << setprecision(2) << (obj / aux.objScale) << ",";
        log.precision(oldPrecision);
    }
    log << timer.elapsedSeconds() << ","
        << mu.physicalMemory << "," << mu.virtualMemory << ","
        << env.randSeed << ","
        << cfg.toBriefStr() << ","
        << generation << "," << iteration << ",";

    // record solution vector.
    for (auto c = output.centers().begin(); c != output.centers().end(); ++c) {
        log << *c << " ";
    }
    log << endl;

    // append all text atomically.
    static mutex logFileMutex;
    lock_guard<mutex> logFileGuard(logFileMutex);

    ofstream logFile(env.logPath, ios::app);
    logFile.seekp(0, ios::end);
    if (logFile.tellp() <= 0) {
        logFile << "Time,ID,Instance,Feasible,ObjMatch,Distance,Duration,PhysMem,VirtMem,RandSeed,Config,Generation,Iteration,Solution" << endl;
    }
    logFile << log.str();
    logFile.close();
    #endif // SZX_DEBUG
}

bool Solver::check(Length &checkerObj) const {
    #if SZX_DEBUG
    enum CheckerFlag {
        IoError = 0x0,
        FormatError = 0x1,
        TooManyCentersError = 0x2
    };

    checkerObj = System::exec("Checker.exe " + env.instPath + " " + env.solutionPathWithTime());
    if (checkerObj > 0) { return true; }
    checkerObj = ~checkerObj;
    if (checkerObj == CheckerFlag::IoError) { Log(LogSwitch::Checker) << "IoError." << endl; }
    if (checkerObj & CheckerFlag::FormatError) { Log(LogSwitch::Checker) << "FormatError." << endl; }
    if (checkerObj & CheckerFlag::TooManyCentersError) { Log(LogSwitch::Checker) << "TooManyCentersError." << endl; }
    return false;
    #else
    checkerObj = 0;
    return true;
    #endif // SZX_DEBUG
}

void Solver::init() {
    ID nodeNum = input.graph().nodenum();

    aux.adjMat.init(nodeNum, nodeNum);
    fill(aux.adjMat.begin(), aux.adjMat.end(), Problem::MaxDistance);
    for (ID n = 0; n < nodeNum; ++n) { aux.adjMat.at(n, n) = 0; }

    if (Problem::isTopologicalGraph(input)) {
        aux.objScale = Problem::TopologicalGraphObjScale;
        for (auto e = input.graph().edges().begin(); e != input.graph().edges().end(); ++e) {
            // only record the last appearance of each edge.
            aux.adjMat.at(e->source(), e->target()) = e->length();
            aux.adjMat.at(e->target(), e->source()) = e->length();
        }

        Timer timer(30s);
        Problem::IsUndirectedGraph
            ? Floyd::findAllPairsPaths_symmetric(aux.adjMat)
            : Floyd::findAllPairsPaths_asymmetric(aux.adjMat);
        Log(LogSwitch::Preprocess) << "Floyd takes " << timer.elapsedSeconds() << " seconds." << endl;
    } else { // geometrical graph.
        aux.objScale = Problem::GeometricalGraphObjScale;
        for (ID n = 0; n < nodeNum; ++n) {
            double nx = input.graph().nodes(n).x();
            double ny = input.graph().nodes(n).y();
            for (ID m = 0; m < n; ++m) {
                Length length = lround(aux.objScale * hypot(
                    nx - input.graph().nodes(m).x(), ny - input.graph().nodes(m).y()));
                aux.adjMat.at(n, m) = length;
                aux.adjMat.at(m, n) = length;
            }
        }
    }

    // load reference results.
    CsvReader cr;
    ifstream ifs(Environment::DefaultInstanceDir() + "Baseline.csv");
    if (!ifs.is_open()) { return; }
    const List<CsvReader::Row> &rows(cr.scan(ifs));
    ifs.close();
    for (auto r = rows.begin(); r != rows.end(); ++r) {
        if (env.friendlyInstName() != r->front()) { continue; }
        aux.refRadius = lround(aux.objScale * stod((*r)[1]));
        break;
    }

    //printInputStatistics();

    //tryBetterRadius();

    // generate ordered adjacency list.
    aux.adjListOrdered.resize(nodeNum);
    for (ID n = 0; n < nodeNum; ++n) {
        List<ID> &adjNodes(aux.adjListOrdered[n]);
        for (ID m = 0; m < nodeNum; ++m) {
            if (aux.adjMat.at(n, m) <= aux.refRadius) { adjNodes.push_back(m); }
        }
        sort(adjNodes.begin(), adjNodes.end(), [&](auto l, auto r) {
            return aux.adjMat.at(n, l) < aux.adjMat.at(n, r);
        });
    }
}

void Solver::printInputStatistics() {
    // count distance distribution.
    Map<Length, ID> distCount; // distCount[l] is the occurance count of distance l.
    for (auto e = aux.adjMat.begin(); e != aux.adjMat.end(); ++e) { ++distCount[*e]; }
    Log(LogSwitch::Preprocess) << distCount.size() << " different distances." << endl;

    // locate the optimal distance and the better one.
    auto refRadius = distCount.find(aux.refRadius);
    auto betterObj = refRadius;
    if (betterObj != distCount.begin()) { --betterObj; }
    Log(LogSwitch::Preprocess) << "distances: " << distCount.begin()->first;
    Log(LogSwitch::Preprocess) << " < ... < " << betterObj->first;
    Log(LogSwitch::Preprocess) << " < " << refRadius->first;
    Log(LogSwitch::Preprocess) << " < " << (++refRadius)->first;
    Log(LogSwitch::Preprocess) << " < ... < " << distCount.rbegin()->first << endl;

    // count covering status.
    Map<Length, ID> coverNodeNums;
    for (ID n = 0; n < input.graph().nodenum(); ++n) {
        ID coverNodeNum = 0;
        for (ID m = 0; m < input.graph().nodenum(); ++m) {
            if (aux.adjMat.at(n, m) <= aux.refRadius) { ++coverNodeNum; }
        }
        ++coverNodeNums[coverNodeNum];
    }

    for (auto n = coverNodeNums.begin(); n != coverNodeNums.end(); ++n) {
        Log(LogSwitch::Preprocess) << n->first << " : " << n->second << endl;
    }
    Log(LogSwitch::Preprocess) << endl;
}

void Solver::tryBetterRadius() {
    Length betterRadius = 0;
    for (auto e = aux.adjMat.begin(); e != aux.adjMat.end(); ++e) {
        if ((*e < aux.refRadius) && (*e > betterRadius)) { betterRadius = *e; }
    }
    aux.refRadius = betterRadius;
    Log(LogSwitch::Preprocess) << "better radius: " << aux.refRadius << endl;
}

bool Solver::optimize(Solution &sln, ID workerId) {
    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " starts." << endl;

    //bool status = optimizePlainModel(sln);
    //bool status = optimizeDecisionModel(sln);
    //bool status = optimizeRelaxedDecisionModel(sln);
    bool status = optimizeCuttOffPMedianModel(sln);

    Log(LogSwitch::Szx::Framework) << "worker " << workerId << " ends." << endl;
    return status;
}

bool Solver::optimizePlainModel(Solution &sln) {
    ID nodeNum = input.graph().nodenum();

    // reset solution state.
    auto &centers(*sln.mutable_centers());
    centers.Reserve(input.centernum());

    MpSolver mp;

    // add decision variables.
    Arr2D<MpSolver::DecisionVar> isServing(nodeNum, nodeNum);
    for (auto x = isServing.begin(); x != isServing.end(); ++x) {
        *x = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
    }

    Arr<MpSolver::DecisionVar> isCenter(nodeNum);
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) {
        *y = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
    }

    MpSolver::DecisionVar maxDist = mp.addVar(MpSolver::VariableType::Real, 0, MpSolver::MaxReal, 0);

    // add constraints.
    // p centers.
    MpSolver::LinearExpr centerNum;
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) { centerNum += *y; }
    //mp.addConstraint(centerNum == input.centernum());
    mp.addConstraint(centerNum <= input.centernum());

    // nodes can only be served by centers.
    for (ID c = 0; c < nodeNum; ++c) {
        for (ID n = 0; n < nodeNum; ++n) {
            mp.addConstraint(isServing.at(c, n) <= isCenter.at(c));
        }
    }

    // each node is served by 1 center only.
    for (ID n = 0; n < nodeNum; ++n) {
        MpSolver::LinearExpr centerNumPerNode;
        for (ID c = 0; c < nodeNum; ++c) {
            centerNumPerNode += isServing.at(c, n);
        }
        //mp.addConstraint(centerNumPerNode == 1);
        mp.addConstraint(centerNumPerNode >= 1);
    }

    // lower bound of the greatest distance.
    for (ID n = 0; n < nodeNum; ++n) {
        MpSolver::LinearExpr distance;
        for (ID c = 0; c < nodeNum; ++c) {
            distance += isServing.at(c, n) * aux.adjMat.at(c, n);
        }
        mp.addConstraint(maxDist >= distance);
    }

    // set objective.
    mp.addObjective(maxDist, MpSolver::OptimaOrientation::Minimize);

    // solve model.
    mp.setOutput(true);

    // record decision.
    if (mp.optimize()) {
        sln.coverRadius = lround(mp.getObjectiveValue());
        for (ID n = 0; n < nodeNum; ++n) {
            if (mp.isTrue(isCenter.at(n))) { centers.Add(n); }
        }
        return true;
    }

    return false;
}

bool Solver::optimizeDecisionModel(Solution &sln) {
    ID nodeNum = input.graph().nodenum();

    // reset solution state.
    auto &centers(*sln.mutable_centers());
    centers.Reserve(input.centernum());

    MpSolver mp;

    // add decision variables.
    Arr<MpSolver::DecisionVar> isCenter(nodeNum);
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) {
        *y = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
    }

    // add constraints.
    // p centers.
    MpSolver::LinearExpr centerNum;
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) { centerNum += *y; }
    //mp.addConstraint(centerNum == input.centernum());
    mp.addConstraint(centerNum <= input.centernum());

    // each node is served by 1 center only.
    for (ID n = 0; n < nodeNum; ++n) {
        MpSolver::LinearExpr centerNumPerNode;
        for (auto c = aux.adjListOrdered[n].begin(); c != aux.adjListOrdered[n].end(); ++c) {
            centerNumPerNode += isCenter.at(*c);
        }
        //mp.addConstraint(centerNumPerNode == 1);
        mp.addConstraint(centerNumPerNode >= 1);
    }

    // solve model.
    mp.setOutput(true);

    // record decision.
    if (mp.optimize()) {
        sln.coverRadius = aux.refRadius;
        for (ID n = 0; n < nodeNum; ++n) {
            if (mp.isTrue(isCenter.at(n))) { centers.Add(n); }
        }
        return true;
    }

    return false;
}

bool Solver::optimizeRelaxedDecisionModel(Solution &sln) {
    ID nodeNum = input.graph().nodenum();

    constexpr double InitNodeWeight = 1;
    constexpr double NodeWeightInc = 0.1;
    List<double> nodeWeights(nodeNum, InitNodeWeight);

    // reset solution state.
    auto &centers(*sln.mutable_centers());
    centers.Reserve(input.centernum());

    MpSolver mp;

    // add decision variables.
    Arr<MpSolver::DecisionVar> isCenter(nodeNum);
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) {
        *y = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
    }

    Arr<MpSolver::DecisionVar> isCovered(nodeNum);
    for (auto x = isCovered.begin(); x != isCovered.end(); ++x) {
        //*x = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
        *x = mp.addVar(MpSolver::VariableType::Real, 0, 1, 0);
    }

    // add constraints.
    // p centers.
    MpSolver::LinearExpr centerNum;
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) { centerNum += *y; }
    //mp.addConstraint(centerNum == input.centernum());
    mp.addConstraint(centerNum <= input.centernum());

    // each node is served by 1 center only.
    for (ID n = 0; n < nodeNum; ++n) {
        MpSolver::LinearExpr centerNumPerNode;
        for (auto c = aux.adjListOrdered[n].begin(); c != aux.adjListOrdered[n].end(); ++c) {
            centerNumPerNode += isCenter.at(*c);
        }
        //mp.addConstraint(centerNumPerNode == 1);
        mp.addConstraint(centerNumPerNode >= isCovered.at(n));
    }

    // number of covered nodes.
    MpSolver::LinearExpr coveredNodeNum;
    for (ID n = 0; n < nodeNum; ++n) {
        coveredNodeNum += (nodeWeights[n] * isCovered.at(n));
    }

    // set objective.
    mp.addObjective(coveredNodeNum, MpSolver::OptimaOrientation::Maximize);

    // solve model.
    mp.setOutput(true);

    //mp.setTimeLimitInSecond(300);

    // record decision.
    if (mp.optimize()) {
        sln.coverRadius = aux.refRadius;
        for (ID n = 0; n < nodeNum; ++n) {
            if (mp.isTrue(isCenter.at(n))) { centers.Add(n); }
        }

        Log(LogSwitch::Szx::Postprocess) << "uncovered nodes under radius " << aux.refRadius << ":";
        for (ID n = 0; n < nodeNum; ++n) {
            if (mp.isTrue(isCovered.at(n))) { continue; }
            nodeWeights[n] += NodeWeightInc;
            Length minDist = Problem::MaxDistance;
            for (auto c = centers.begin(); c != centers.end(); ++c) {
                minDist = min(aux.adjMat.at(n, *c), minDist);
            }
            if (sln.coverRadius < minDist) { sln.coverRadius = minDist; }
            Log(LogSwitch::Szx::Postprocess) << " " << n << "(" << minDist << ")";
        }
        Log(LogSwitch::Szx::Postprocess) << endl;

        return true;
    }

    return false;
}

bool Solver::optimizeCuttOffPMedianModel(Solution &sln) {
    ID nodeNum = input.graph().nodenum();

    // reset solution state.
    auto &centers(*sln.mutable_centers());
    centers.Reserve(input.centernum());

    MpSolver mp;

    // add decision variables.
    Arr2D<MpSolver::DecisionVar> isServing(nodeNum, nodeNum);
    for (auto x = isServing.begin(); x != isServing.end(); ++x) {
        *x = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
    }

    Arr<MpSolver::DecisionVar> isCenter(nodeNum);
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) {
        *y = mp.addVar(MpSolver::VariableType::Bool, 0, 1, 0);
    }

    // add constraints.
    // p centers.
    MpSolver::LinearExpr centerNum;
    for (auto y = isCenter.begin(); y != isCenter.end(); ++y) { centerNum += *y; }
    //mp.addConstraint(centerNum == input.centernum());
    mp.addConstraint(centerNum <= input.centernum());

    // nodes can only be served by centers.
    for (ID c = 0; c < nodeNum; ++c) {
        for (ID n = 0; n < nodeNum; ++n) {
            mp.addConstraint(isServing.at(c, n) <= isCenter.at(c));
        }
    }

    // each node is served by 1 center only.
    for (ID n = 0; n < nodeNum; ++n) {
        MpSolver::LinearExpr centerNumPerNode;
        for (ID c = 0; c < nodeNum; ++c) {
            centerNumPerNode += isServing.at(c, n);
        }
        //mp.addConstraint(centerNumPerNode == 1);
        mp.addConstraint(centerNumPerNode >= 1);
    }

    // distance to uncovered nodes.
    MpSolver::LinearExpr distance;
    for (ID n = 0; n < nodeNum; ++n) {
        for (ID c = 0; c < nodeNum; ++c) {
            Length dist = max(aux.adjMat.at(c, n) - aux.refRadius, 0);
            distance += isServing.at(c, n) * dist;
        }
    }

    // set objective.
    mp.addObjective(distance, MpSolver::OptimaOrientation::Minimize);

    // solve model.
    mp.setOutput(true);

    //mp.setTimeLimitInSecond(300);

    // record decision.
    if (mp.optimize()) {
        sln.coverRadius = aux.refRadius;
        for (ID n = 0; n < nodeNum; ++n) {
            if (mp.isTrue(isCenter.at(n))) { centers.Add(n); }
        }
        return true;
    }

    return false;
}
#pragma endregion Solver

}
