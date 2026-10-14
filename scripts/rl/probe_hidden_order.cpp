// Standalone diagnostic linked to existing native objects; no engine edits.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <vector>
#include "combat/BattleContext.h"
#include "game/GameContext.h"
#include "game/Card.h"
#include "sim/search/BattleScumSearcher2.h"
#include "json.hpp"

using namespace sts;
using nlohmann::json;

auto publicKey(const CardInstance &c) {
    return std::make_tuple(static_cast<int>(c.id), c.getUpgradeCount(), c.cost,
                           c.costForTurn, c.freeToPlayOnce, c.retain);
}

void canonicalize(BattleContext &bc) {
    auto less=[](const CardInstance &a,const CardInstance &b){return publicKey(a)<publicKey(b);};
    std::sort(bc.cards.drawPile.begin(),bc.cards.drawPile.end(),less);
    std::sort(bc.cards.discardPile.begin(),bc.cards.discardPile.end(),less);
    std::sort(bc.cards.exhaustPile.begin(),bc.cards.exhaustPile.end(),less);
}

std::map<std::uint32_t,int> votes(const BattleContext &original,int samples,int simulations,bool canonical) {
    std::map<std::uint32_t,int> out;
    for(int sample=1;sample<=samples;++sample) {
        BattleContext bc(original);
        if(canonical)canonicalize(bc);
        std::uint64_t seed=173*1009+sample;
        bc.aiRng=Random(seed*6+1);bc.cardRandomRng=Random(seed*6+2);bc.miscRng=Random(seed*6+3);
        bc.monsterHpRng=Random(seed*6+4);bc.potionRng=Random(seed*6+5);bc.shuffleRng=Random(seed*6+6);
        std::mt19937_64 drawRng(sample);
        std::shuffle(bc.cards.drawPile.begin(),bc.cards.drawPile.end(),drawRng);
        search::BattleScumSearcher2 searcher(bc);
        searcher.randGen.seed(173+sample);searcher.search(simulations);
        const search::Action *selected=nullptr;
        if(searcher.outcomePlayerHp>0 && !searcher.bestActionSequence.empty()
                && searcher.bestActionSequence.front().isValidAction(original)) {
            selected=&searcher.bestActionSequence.front();
        } else {
            std::int64_t maximum=-1;
            for(const auto &edge:searcher.root.edges) {
                if(edge.action.isValidAction(original) && edge.node.simulationCount>maximum) {
                    selected=&edge.action;maximum=edge.node.simulationCount;
                }
            }
        }
        if(!selected)throw std::runtime_error("No legal root edge");
        ++out[selected->bits];
    }
    return out;
}

std::uint32_t plurality(const std::map<std::uint32_t,int> &v) {
    // Python preserves sample insertion order on ties; this diagnostic compares
    // histograms directly, and uses lowest bits only as a secondary summary.
    return std::max_element(v.begin(),v.end(),[](const auto &a,const auto &b){return a.second<b.second;})->first;
}

int main(int argc,char **argv) {
    const int worlds=argc>1?std::stoi(argv[1]):16;
    const int samples=argc>2?std::stoi(argv[2]):24;
    const int simulations=argc>3?std::stoi(argv[3]):100;
    for(int i=0;i<worlds;++i) {
        GameContext gc(CharacterClass::IRONCLAD,43000000000ULL+i,0);
        gc.act=2;gc.floorNum=33;gc.curRoom=Room::BOSS;
        for(auto card:{CardId::POMMEL_STRIKE,CardId::SHRUG_IT_OFF,CardId::UPPERCUT,
                      CardId::INFLAME,CardId::FLAME_BARRIER,CardId::ANGER}) gc.deck.obtain(gc,Card(card));
        BattleContext original;original.init(gc,MonsterEncounter::CHAMP);original.executeActions();
        if(original.inputState!=InputState::PLAYER_NORMAL || original.outcome!=Outcome::UNDECIDED)
            throw std::runtime_error("Constructed start did not reach a player decision");
        BattleContext reversed(original);
        std::reverse(reversed.cards.drawPile.begin(),reversed.cards.drawPile.end());
        std::reverse(reversed.cards.discardPile.begin(),reversed.cards.discardPile.end());
        std::reverse(reversed.cards.exhaustPile.begin(),reversed.cards.exhaustPile.end());
        auto left=votes(original,samples,simulations,false),right=votes(reversed,samples,simulations,false);
        auto cleft=votes(original,samples,simulations,true),cright=votes(reversed,samples,simulations,true);
        std::cout<<json{{"seed",gc.seed},{"samples",samples},{"simulations",simulations},
                       {"draw_cards",original.cards.drawPile.size()},
                       {"votes_original",left},{"votes_reversed",right},
                       {"votes_differ",left!=right},{"plurality_differ",plurality(left)!=plurality(right)},
                       {"canonical_votes_equal",cleft==cright}}
                 .dump()<<std::endl;
    }
}
