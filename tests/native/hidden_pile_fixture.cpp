// Test-only mutation of scratch native copies; never installed as a policy API.
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "combat/BattleContext.h"
namespace py=pybind11;
PYBIND11_MODULE(sts_hidden_pile_fixture,m) {
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
