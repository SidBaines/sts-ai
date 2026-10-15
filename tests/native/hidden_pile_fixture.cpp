// Test-only helpers return modified native copies; never installed as a policy API.
#include <algorithm>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "combat/BattleContext.h"
#include "game/GameContext.h"
namespace py=pybind11;
PYBIND11_MODULE(sts_hidden_pile_fixture,m) {
    m.def("encounter_game_copy",[](const sts::GameContext &gc,
                                   sts::MonsterEncounter encounter) {
        sts::GameContext result(gc);
        result.info.encounter = encounter;
        return result;
    });
    m.def("monster_hp_copy",[](const sts::BattleContext &bc, int targetIdx, int hp) {
        if (targetIdx < 0 || targetIdx >= bc.monsters.monsterCount
                || hp <= 0 || hp > bc.monsters.arr[targetIdx].maxHp
                || bc.monsters.arr[targetIdx].curHp <= 0) {
            throw std::runtime_error("invalid monster HP setup");
        }
        sts::BattleContext result(bc);
        result.monsters.arr[targetIdx].curHp = hp;
        return result;
    });
    m.def("combat_hit_setup_copy",[](const sts::BattleContext &bc, int targetIdx,
                                     int block, int energy, int strength) {
        if (targetIdx < 0 || targetIdx >= bc.monsters.monsterCount || block < 0) {
            throw std::runtime_error("invalid combat hit setup");
        }
        sts::BattleContext result(bc);
        result.monsters.arr[targetIdx].block = block;
        result.player.energy = energy;
        result.player.strength = strength;
        return result;
    });
    m.def("copy",[](const sts::BattleContext &bc) {
        return sts::BattleContext(bc);
    });
    m.def("damage_annotations_copy",[](const sts::BattleContext &bc, int energy,
                                       int strength, int weak,
                                       const std::vector<int> &vulnerable,
                                       bool sameName) {
        if (energy < 0 || strength < 0 || weak < 0
                || vulnerable.size() != static_cast<std::size_t>(bc.monsters.monsterCount)
                || std::any_of(vulnerable.begin(), vulnerable.end(), [](int n) { return n < 0; })) {
            throw std::runtime_error("invalid damage annotation fixture parameters");
        }
        sts::BattleContext result(bc);
        result.player.energy = energy;
        result.player.setStatusValueNoChecks<::PlayerStatus::STRENGTH>(strength);
        result.player.debuff<::PlayerStatus::WEAK>(weak, false);
        for (int i = 0; i < result.monsters.monsterCount; ++i) {
            auto &monster = result.monsters.arr[i];
            if (sameName) monster.id = result.monsters.arr[0].id;
            // Prevent overkill and reactive powers (e.g. Curl Up) from hiding
            // the pre-block damage that these tests compare with the annotation.
            monster.curHp = monster.maxHp = 500;
            monster.block = 0;
            monster.statusBits = 0;
            monster.setStatus<sts::MonsterStatus::VULNERABLE>(vulnerable[i]);
            monster.setHasStatus<sts::MonsterStatus::VULNERABLE>(vulnerable[i] > 0);
        }
        return result;
    });
    m.def("hand_flags_copy",[](const sts::BattleContext &bc, int handIdx,
                                bool freeOnce, bool retain) {
        if (handIdx < 0 || handIdx >= bc.cards.cardsInHand) {
            throw std::runtime_error("hand index out of range");
        }
        sts::BattleContext result(bc);
        result.cards.hand[handIdx].freeToPlayOnce = freeOnce;
        result.cards.hand[handIdx].retain = retain;
        return result;
    });
    m.def("reversed_copy",[](const sts::BattleContext &bc) {
        sts::BattleContext result(bc);
        std::reverse(result.cards.drawPile.begin(),result.cards.drawPile.end());
        std::reverse(result.cards.discardPile.begin(),result.cards.discardPile.end());
        std::reverse(result.cards.exhaustPile.begin(),result.cards.exhaustPile.end());
        return result;
    });
    m.def("pile_selection_copy",[](const sts::BattleContext &bc, bool exhaust) {
        sts::BattleContext result(bc);
        auto &target=exhaust?result.cards.exhaustPile:result.cards.discardPile;
        target.insert(target.end(),result.cards.drawPile.begin(),result.cards.drawPile.end());
        result.cards.drawPile.clear();
        result.inputState=sts::InputState::CARD_SELECT;
        result.cardSelectInfo.cardSelectTask=exhaust?sts::CardSelectTask::EXHUME:sts::CardSelectTask::HEADBUTT;
        result.cardSelectInfo.pickCount=1;
        return result;
    });
    m.def("draw_ids",[](const sts::BattleContext &bc) {
        std::vector<int> ids;
        for(const auto &card:bc.cards.drawPile)ids.push_back(static_cast<int>(card.id));
        return ids;
    });
}
