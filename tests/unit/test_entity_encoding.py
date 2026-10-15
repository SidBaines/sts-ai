"""Synthetic and captured env-v0.1.5 public fixtures; no simulator needed."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from sts_policy import entity_encoding
from sts_policy.entity_encoding import (
    ACTION_TYPES, ENTITY_TYPES, ENCODER_SHA256, ENCODER_VERSION, PAD, UNK, VOCABULARY_SHA256,
    EncodedEntityObservation, collate_observations, encode_observation, vocabulary_id,
)


def card(name='Strike', index=0, cost=1, current=1, free=False, retain=False, count=None):
    upgrade = re.search(r'\+(\d*)$', name)
    c = {'name': name, 'label': f'{name} [Attack] (cost {current})', 'type': 'Attack',
         'upgrade_count': int(upgrade[1] or 1) if upgrade else 0,
         'cost': cost, 'cost_for_turn': current, 'free_to_play_once': free, 'retain': retain}
    if index is not None:
        c['index'] = index
    if count is not None:
        c['count'] = count
    return c


def combat_observation():
    # publicCardDict/publicPlayer/enemies/publicPotions; action labels from
    # describeBattleAction, appendEnemyLabel and CombatPublicV5Env.selection_label.
    return {'phase': 'combat', 'text': 'Battle turn 2\nPlayer HP: 51/80',
        'state': {'act': 1, 'floor': 7, 'cur_hp': 51, 'max_hp': 80, 'gold': 99, 'ascension': 3,
            'phase': 'combat', 'room': 'Room.ELITE', 'screen_state': 'ScreenState.BATTLE',
            'keys': {'ruby': False, 'emerald': True, 'sapphire': False},
            'combat': {'turn': 2, 'player_cur_hp': 51, 'player_max_hp': 80, 'player_energy': 3,
                'player': {'cur_hp': 51, 'max_hp': 80, 'gold': 99, 'block': 7, 'energy': 3,
                    'energy_per_turn': 3, 'card_draw_per_turn': 5, 'orb_slots': 0, 'stance': 'NEUTRAL',
                    'turn': 2, 'cards_played_this_turn': 2, 'attacks_played_this_turn': 1,
                    'skills_played_this_turn': 1, 'cards_discarded_this_turn': 0,
                    'powers': {'Strength': 2, 'Weak': 1, 'The Bomb (2 turns)': 40}},
                'enemies': [
                    {'index': 0, 'name': 'SENTRY', 'cur_hp': 8, 'max_hp': 40, 'block': 2,
                     'intent': 'SENTRY_BEAM', 'intent_damage': 9, 'intent_hits': 1, 'alive': True,
                     'strength': 0, 'vulnerable': 2, 'weak': 0, 'poison': 0, 'powers': {'Artifact': 1}},
                    {'index': 1, 'name': 'SENTRY', 'cur_hp': 35, 'max_hp': 40, 'block': 0,
                     'intent': 'SENTRY_BOLT', 'intent_damage': 0, 'intent_hits': -1, 'alive': True,
                     'strength': 3, 'vulnerable': 0, 'weak': 1, 'poison': 0, 'powers': {'Artifact': 2}}],
                'cards': {'hand': [card(index=0), card(index=1, free=True), card(index=2, current=0),
                                   card(index=3, retain=True), card('Strike+', index=4), card(index=5)],
                          'draw': [card(index=None, count=2), card('Strike+', index=None, count=1)],
                          'discard': [card(index=None, current=0, free=True, count=3)], 'exhaust': []},
                'relics': [{'name': 'Pen Nib', 'counter': 5, 'counter_is_live': True},
                           {'name': 'Burning Blood', 'counter': 0, 'counter_is_live': False}],
                'potions': {'capacity': 3, 'occupied': 2, 'slots': [
                    {'index': 0, 'empty': False, 'name': 'Fire Potion'},
                    {'index': 1, 'empty': True, 'name': None},
                    {'index': 2, 'empty': False, 'name': 'Block Potion'}]},
                'recent_actions': ['play Bash [Attack] (cost 2) -> SENTRY [enemy 0] (deal 8)',
                                   'play Defend [Skill] (cost 1)']}},
        'legal_actions': [
            {'index': 10, 'description': 'play Strike [Attack] (cost 1) -> SENTRY [enemy 0] (deal 9)'},
            {'index': 11, 'description': 'play Strike [Attack] (cost 1) -> SENTRY [enemy 1] (deal 6)'},
            {'index': 12, 'description': 'play Strike [Attack] (cost 1, free once) -> SENTRY [enemy 0] (deal 9)'},
            {'index': 13, 'description': 'play Strike [Attack] (cost 0) -> SENTRY [enemy 0] (deal 9)'},
            {'index': 14, 'description': 'play Strike [Attack] (cost 1, retained) -> SENTRY [enemy 0] (deal 9)'},
            {'index': 15, 'description': 'play Strike+ [Attack] (cost 1) -> SENTRY [enemy 0] (deal 12)'},
            {'index': 16, 'description': 'drink potion Fire Potion -> SENTRY [enemy 1]'},
            {'index': 17, 'description': 'end turn'}]}


def ooc_observation():
    # describeGameState and describeGameAction use Deck::operator<<,
    # RelicContainer::operator<< and slots [0, potionCount), including holes.
    return {'phase': 'out_of_combat',
        'state': {'act': 2, 'floor': 20, 'cur_hp': 60, 'max_hp': 88, 'gold': 200, 'ascension': 5,
                  'keys': {'ruby': True, 'emerald': False, 'sapphire': True},
                  'screen_state': 'ScreenState.CARD_SELECT', 'room': 'Room.REST',
                  'event': {'event': 'FALLING', 'falling_attack_card': 'Strike+', 'phase': 1}},
        'text': 'Act 2, floor 20, screen CARD_SELECT, room REST, end-of-act boss CHAMP\n'
                'Your HP: 60/88, gold: 200\nDeck: (4): {Strike, Strike+, Searing Blow+2, Defend,}\n'
                'Relics: {Burning Blood:0, Pen Nib:7,}\nPotions: Blood Potion, EMPTY_POTION_SLOT\n'
                'Choose a card to UPGRADE (chosen 0 of 1).',
        'legal_actions': [
            {'index': 0, 'description': 'select card index 1: Strike+ [Attack, Basic]'},
            {'index': 1, 'description': 'drink potion Blood Potion (slot 0): heal 17 HP'},
            {'index': 2, 'description': 'discard potion Blood Potion (slot 0)'},
            {'index': 3, 'description': 'discard potion Fire Potion (slot 2)'}]}


def array_digest(encoded):
    digest = hashlib.sha256()
    for f in fields(EncodedEntityObservation):
        value = getattr(encoded, f.name)
        if isinstance(value, np.ndarray):
            digest.update(f.name.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    digest.update(json.dumps(encoded.coverage, sort_keys=True).encode())
    return digest.hexdigest()


class EntityEncodingTests(unittest.TestCase):
    def test_encoder_sha_binds_installed_source_bytes(self):
        source = Path(entity_encoding.__file__).read_bytes()
        self.assertEqual(ENCODER_SHA256, hashlib.sha256(source).hexdigest())

    def test_contract_slots_duplicates_and_intents(self):
        e = encode_observation(combat_observation())
        self.assertEqual(ENCODER_VERSION, 'public_entities_fresh_v1')
        self.assertEqual(e.entity_numeric.shape, (len(e.entity_ids), 32))
        self.assertEqual(e.action_numeric.shape, (8, 32))
        self.assertEqual(e.action_text.shape, (8, 32))
        self.assertEqual(e.action_indices.dtype, np.int64)
        self.assertEqual(e.action_indices.tolist(), list(range(10, 18)))
        hand = np.flatnonzero(e.entity_types == ENTITY_TYPES['hand'])
        enemies = np.flatnonzero(e.entity_types == ENTITY_TYPES['enemy'])
        self.assertEqual(e.action_source[:6].tolist(), [hand[0], hand[0], hand[1], hand[2], hand[3], hand[4]])
        self.assertEqual(e.action_target[:2].tolist(), enemies.tolist())
        self.assertEqual(e.action_target[6], enemies[1])
        self.assertNotEqual(e.entity_aux[enemies[0], 0], e.entity_aux[enemies[1], 0])
        self.assertEqual(e.entity_aux[enemies[0], 0], vocabulary_id('intent', 'SENTRY_BEAM'))
        self.assertEqual(e.coverage['unk_by_namespace'], {})
        for array in (e.entity_numeric, e.action_numeric, e.action_text, e.power_amounts):
            self.assertTrue(np.isfinite(array).all())
        self.assertEqual(e.entity_ids.dtype, np.int64)

    def test_private_fields_ignored_and_menu_indices_only_metadata(self):
        a = combat_observation()
        b = deepcopy(a)
        b['info'] = {'seed': 124, 'native': object(), 'future_value': 999}
        b['state'].update(world_seed=123, seed=5, private_card_order=['Secret'], native_values=[9])
        b['state']['combat'].update(search_values=[9], rng_state='hidden')
        b['state']['combat']['player']['rng_state'] = 'secret'
        b['state']['combat']['cards']['hand'][0]['bits'] = 999
        b['text'] += '\nprivate_diagnostic: seed 123 hidden value 999'
        for action in b['legal_actions']:
            action.update(bits=123456, native=object(), future_value=100)
        self.assertEqual(array_digest(encode_observation(a)), array_digest(encode_observation(b)))
        for action in b['legal_actions']:
            action['index'] += 100
        ea, eb = encode_observation(a), encode_observation(b)
        ba, bb = collate_observations([ea]), collate_observations([eb])
        self.assertNotIn('action_indices', bb)
        for key in ba:
            self.assertTrue(ba[key].equal(bb[key]), key)

    def test_powers_keep_owner_and_amounts(self):
        e = encode_observation(combat_observation())
        enemies = np.flatnonzero(e.entity_types == ENTITY_TYPES['enemy'])
        artifact = e.power_ids == vocabulary_id('power', 'Artifact')
        self.assertEqual(e.power_owner[artifact].tolist(), enemies.tolist())
        np.testing.assert_allclose(e.power_amounts[artifact], [.01, .02])
        player = np.flatnonzero(e.entity_types == ENTITY_TYPES['player'])[0]
        strength = e.power_ids == vocabulary_id('power', 'Strength')
        self.assertIn(player, e.power_owner[strength])

    def test_invalid_power_amount_retains_identity_with_availability(self):
        obs = combat_observation()
        obs['state']['combat']['player']['powers']['Strength'] = float('inf')
        e = encode_observation(obs)
        player = np.flatnonzero(e.entity_types == ENTITY_TYPES['player'])[0]
        selected = (e.power_ids == vocabulary_id('power', 'Strength')) & (e.power_owner == player)
        self.assertEqual(selected.sum(), 1)
        self.assertEqual(e.power_amounts[selected][0], 0)
        self.assertFalse(e.power_amount_available[selected][0])
        self.assertEqual(e.coverage['parser_fallbacks'], 1)

    def test_multisets_permutation_and_count_preservation(self):
        obs = combat_observation()
        other = deepcopy(obs)
        cards = other['state']['combat']['cards']
        cards['draw'].reverse()
        other['state']['combat']['relics'].reverse()
        # A split group and a pre-aggregated group encode identically.
        cards['discard'] = [card(index=None, current=0, free=True), card(index=None, current=0, free=True, count=2)]
        a, b = encode_observation(obs), encode_observation(other)
        self.assertEqual(array_digest(a), array_digest(b))
        draw = a.entity_types == ENTITY_TYPES['draw']
        # Header + distinct upgraded/unupgraded groups, counts preserved.
        self.assertEqual(a.coverage['draw_count'], 3)
        self.assertEqual(a.coverage['discard_count'], 3)
        np.testing.assert_allclose(a.entity_numeric[draw, 0].sum(), .06)
        other['state']['combat']['cards']['draw'][0]['count'] += 1
        self.assertNotEqual(array_digest(a), array_digest(encode_observation(other)))

    def test_pile_mechanical_variants_and_unavailable_values(self):
        obs = combat_observation()
        obs['state']['combat']['cards']['draw'] = [card(index=None), card(index=None, free=True),
            card(index=None, retain=True), card(index=None, current=0), card('Strike+', index=None)]
        e = encode_observation(obs)
        self.assertEqual(sum(e.entity_types == ENTITY_TYPES['draw']), 6)  # total + groups
        obs['state']['combat']['player'].pop('card_draw_per_turn')
        obs['state']['combat']['player']['block'] = float('nan')
        missing = encode_observation(obs)
        player = np.flatnonzero(missing.entity_types == ENTITY_TYPES['player'])[0]
        self.assertEqual(missing.entity_numeric[player, 16 + 5], 0)
        self.assertEqual(missing.entity_numeric[player, 16 + 2], 0)
        self.assertEqual(missing.entity_numeric[player, 2], 0)
        obs['state']['combat']['player']['block'] = 0
        zero = encode_observation(obs)
        self.assertEqual(zero.entity_numeric[player, 16 + 2], 1)

    def test_base_cost_ambiguity_is_not_first_name_match(self):
        obs = combat_observation()
        obs['state']['combat']['cards']['hand'][5]['cost'] = 2
        e = encode_observation(obs)
        self.assertEqual(e.action_source[0], -1)
        self.assertEqual(e.coverage['ambiguous_sources'], 2)
        self.assertGreater(e.action_source[2], -1)  # free-once remains resolvable
        # Pile selection labels show base cost, including the implicit equal-
        # to-current case; they can distinguish what play labels cannot.
        obs['state']['combat']['cards']['draw'] = [card(index=None, cost=1), card(index=None, cost=2)]
        obs['legal_actions'] = [{'index': 0, 'description': 'select card for SECRET_WEAPON: Strike [Attack] (cost 1)'},
                                {'index': 1, 'description': 'select card for SECRET_WEAPON: Strike [Attack] (cost 1, base 2)'}]
        e = encode_observation(obs)
        self.assertNotEqual(e.action_source[0], e.action_source[1])
        self.assertTrue((e.action_source >= 0).all())

    def test_native_counter_names_in_hand_actions_and_recent_plays(self):
        obs = combat_observation()
        # env-v0.1.5 publicCardName: =specialData BEFORE the upgrade '+'.
        names = ['Rampage=8+', 'Rampage=16+', 'Ritual Dagger=23', 'Genetic Algorithm=5+', 'Strike']
        types = ['Attack', 'Attack', 'Attack', 'Skill', 'Attack']
        obs['state']['combat']['cards']['hand'] = [
            {**card(name, index=i), 'type': kind} for i, (name, kind) in enumerate(zip(names, types))]
        labels = [f'play {name} [{kind}] (cost 1)' +
                  (' -> SENTRY [enemy 0] (deal 13)' if kind == 'Attack' else '')
                  for name, kind in zip(names, types)]
        obs['legal_actions'] = [{'index': i, 'description': s} for i, s in enumerate(labels)]
        obs['state']['combat']['recent_actions'] = labels
        e = encode_observation(obs)
        hand = np.flatnonzero(e.entity_types == ENTITY_TYPES['hand'])
        expected = [vocabulary_id('card', n) for n in ('Rampage', 'Rampage', 'Ritual Dagger', 'Genetic Algorithm', 'Strike')]
        self.assertNotIn(UNK, expected)
        self.assertEqual(e.action_source.tolist(), hand.tolist())
        for actual in (e.entity_ids[hand], e.action_ids, e.recent_ids):
            self.assertEqual(actual.tolist(), expected)
        for actual in (e.entity_numeric[hand], e.action_numeric, e.recent_numeric):
            np.testing.assert_allclose(actual[:, 11], [.08, .16, .23, .05, 0])
            np.testing.assert_array_equal(actual[:, 27], [1, 1, 1, 1, 0])
            np.testing.assert_allclose(actual[:, 1], [.1, .1, 0, .1, 0])
        self.assertEqual(e.coverage['unk_by_namespace'], {})
        self.assertEqual(e.coverage['unresolved_sources'], 0)
        self.assertEqual(e.coverage['ambiguous_sources'], 0)
        for name in names:
            self.assertNotEqual(vocabulary_id('card', name), UNK)

    def test_native_counter_pile_groups_and_deck_selection(self):
        obs = combat_observation()
        variants = [card('Rampage=8', index=None, count=2), card('Rampage=8', index=None),
                    card('Rampage=16', index=None), card('Rampage=8+', index=None),
                    {**card('Genetic Algorithm=0+', index=None), 'type': 'Skill'}]
        for pile in ('draw', 'discard', 'exhaust'):
            obs['state']['combat']['cards'][pile] = deepcopy(variants)
        obs['legal_actions'] = [{'index': i, 'description': s} for i, s in enumerate([
            'select card for HEADBUTT: Rampage=8 [Attack] (cost 1)',
            'select card for HEADBUTT: Rampage=16 [Attack] (cost 1)',
            'select card for EXHUME: Rampage=8+ [Attack] (cost 1)',
            'select card for SECRET_WEAPON: Rampage=16 [Attack] (cost 1)'])]
        e = encode_observation(obs)
        self.assertEqual(e.coverage['unresolved_sources'], 0)
        self.assertEqual(e.coverage['ambiguous_sources'], 0)
        self.assertNotEqual(e.action_source[0], e.action_source[1])
        for pile in ('draw', 'discard', 'exhaust'):
            entities = np.flatnonzero((e.entity_types == ENTITY_TYPES[pile]) &
                                      (e.entity_ids != vocabulary_id('role', pile)))
            self.assertEqual(len(entities), 4)
            self.assertEqual(e.coverage[pile + '_count'], 6)
            self.assertTrue((e.entity_numeric[entities, 27] == 1).all())
            np.testing.assert_allclose(sorted(e.entity_numeric[entities, 0]), [.01, .01, .01, .03])
        reversed_obs = deepcopy(obs)
        for pile in ('draw', 'discard', 'exhaust'):
            reversed_obs['state']['combat']['cards'][pile].reverse()
        self.assertEqual(array_digest(e), array_digest(encode_observation(reversed_obs)))

        # OOC PrintHelpers::operator<<(Card) exposes =misc for Ritual Dagger.
        obs = ooc_observation()
        obs['text'] = ('Deck: (3): {Ritual Dagger=15+, Ritual Dagger=15+, Ritual Dagger=20+,}\n'
                       'Choose a card to REMOVE from your deck (chosen 0 of 1).')
        obs['state']['event'] = {'event': 'WE_MEET_AGAIN', 'card': 'Ritual Dagger=31+'}
        obs['legal_actions'] = [{'index': i, 'description': f'select card index {i}: Ritual Dagger={amount}+ [Attack, Special]'}
                                for i, amount in enumerate((15, 20))]
        e = encode_observation(obs)
        self.assertTrue((e.action_source >= 0).all())
        self.assertNotEqual(e.action_source[0], e.action_source[1])
        np.testing.assert_allclose(e.entity_numeric[e.action_source, 0], [.02, .01])
        np.testing.assert_allclose(e.entity_numeric[e.action_source, 11], [.15, .20])
        np.testing.assert_allclose(e.action_numeric[:, 11], [.15, .20])
        self.assertTrue((e.action_numeric[:, 27] == 1).all())
        offer = np.flatnonzero(e.entity_types == ENTITY_TYPES['offer'])
        self.assertEqual(e.entity_ids[offer[0]], vocabulary_id('card', 'Ritual Dagger'))
        self.assertEqual(e.entity_numeric[offer[0], 11], np.float32(.31))
        self.assertEqual(e.coverage['unk_by_namespace'], {})

    def test_native_unplayable_cost_class_preserves_types_and_source_joins(self):
        obs = combat_observation()
        curse = {**card("Ascender's Bane", cost=-3, current=-3), 'type': 'Curse'}
        status = {**card('Dazed', index=1, cost=-2, current=-2), 'type': 'Status'}
        xcost = card('Whirlwind', index=2, cost=-1, current=-1)
        changed = {**card("Ascender's Bane", index=3, cost=-3, current=0, free=True), 'type': 'Curse'}
        obs['state']['combat']['cards']['hand'] = [curse, status, xcost, changed]
        for pile in ('draw', 'discard', 'exhaust'):
            obs['state']['combat']['cards'][pile] = [dict(curse, count=2), dict(status, count=1), dict(xcost, count=1), dict(changed, count=1)]
        labels = ["select card for EXHAUST_ONE: Ascender's Bane [Curse] (cost unplayable)",
                  'select card for WARCRY: Dazed [Status] (cost unplayable)',
                  'select card for FORETHOUGHT: Whirlwind [Attack] (cost X)',
                  "select card for EXHAUST_ONE: Ascender's Bane [Curse] (cost 0, base unplayable, free once)",
                  "select card for HEADBUTT: Ascender's Bane [Curse] (cost unplayable)",
                  'select card for HOLOGRAM: Dazed [Status] (cost unplayable)',
                  "select card for SECRET_TECHNIQUE: Ascender's Bane [Curse] (cost unplayable)",
                  "select card for EXHUME: Ascender's Bane [Curse] (cost unplayable)"]
        obs['legal_actions'] = [{'index': i, 'description': label} for i, label in enumerate(labels)]
        e = encode_observation(obs)
        hand = np.flatnonzero(e.entity_types == ENTITY_TYPES['hand'])
        self.assertEqual(e.action_source[:4].tolist(), hand.tolist())
        self.assertEqual(e.entity_types[e.action_source[4:]].tolist(),
                         [ENTITY_TYPES[p] for p in ('discard', 'discard', 'draw', 'exhaust')])
        self.assertEqual(e.coverage['unresolved_sources'], 0)
        self.assertEqual(e.coverage['ambiguous_sources'], 0)
        self.assertEqual(e.coverage['unk_by_namespace'], {})
        np.testing.assert_allclose(e.action_numeric[:, 2], [-.4, -.4, -.2, -.4, -.4, -.4, -.4, -.4])
        np.testing.assert_allclose(e.action_numeric[:4, 3], [-.4, -.4, -.2, 0])
        np.testing.assert_array_equal(e.action_numeric[:4, 4], [0, 0, 0, 1])
        np.testing.assert_allclose(e.entity_numeric[e.action_source, 2:4], e.action_numeric[:, 2:4])
        np.testing.assert_array_equal(e.entity_aux[e.action_source, 0], e.action_aux[:, 0])
        self.assertEqual(e.action_aux[0, 0], vocabulary_id('card_type', 'Curse'))
        self.assertEqual(e.action_aux[1, 0], vocabulary_id('card_type', 'Status'))

    def test_multi_selection_and_pile_selection(self):
        obs = combat_observation()
        obs['legal_actions'] = [{'index': i, 'description': text} for i, text in enumerate([
            'exhaust: hand [0] Strike [Attack] (cost 1); hand [1] Strike [Attack] (cost 1, free once)',
            'exhaust no cards', 'keep current hand (discard no cards)',
            'discard and redraw: hand [3] Strike [Attack] (cost 1, retained)',
            'select card for HEADBUTT: Strike [Attack] (cost 0, base 1, free once)',
            'select card for SECRET_WEAPON: Strike+ [Attack] (cost 1)',
            'select card for CODEX: Bash [Attack] (cost 2)', 'select card for CODEX: skip'])]
        e = encode_observation(obs)
        self.assertEqual(e.action_source_mask.sum(1).tolist(), [2, 0, 0, 1, 1, 1, 0, 0])
        self.assertEqual(e.entity_types[e.action_source[4]], ENTITY_TYPES['discard'])
        self.assertEqual(e.entity_types[e.action_source[5]], ENTITY_TYPES['draw'])
        self.assertEqual(e.action_ids[6], vocabulary_id('card', 'Bash'))
        self.assertEqual(e.action_types[-1], ACTION_TYPES['skip_card'])

    def test_ooc_inventory_upgrades_potion_slots_and_context(self):
        e = encode_observation(ooc_observation())
        self.assertEqual(e.coverage['deck_count'], 4)
        self.assertEqual(e.coverage['potion_count'], 2)
        self.assertEqual(e.entity_types[e.action_source[0]], ENTITY_TYPES['deck'])
        self.assertEqual(e.entity_numeric[e.action_source[0], 1], np.float32(.1))
        potions = np.flatnonzero(e.entity_types == ENTITY_TYPES['potion'])
        np.testing.assert_allclose(e.entity_numeric[potions, 0], [0, .1, .2])
        np.testing.assert_array_equal(e.entity_numeric[potions, 1], [0, 1, 0])
        self.assertTrue((e.entity_numeric[potions, 16] == 1).all())
        self.assertEqual(e.action_source[1], e.action_source[2])
        self.assertNotEqual(e.action_source[2], e.action_source[3])
        searing = np.flatnonzero(e.entity_ids == vocabulary_id('card', 'Searing Blow'))
        self.assertEqual(e.entity_numeric[searing[0], 1], np.float32(.2))
        context = np.flatnonzero(e.entity_types == ENTITY_TYPES['context'])
        self.assertIn(vocabulary_id('purpose', 'UPGRADE'), e.entity_aux[context, 1])

    def test_inventory_missing_empty_malformed_and_unknown(self):
        base = {'phase': 'out_of_combat', 'state': {}, 'text': '',
                'legal_actions': [{'index': 0, 'description': 'leave shop'}]}
        self.assertEqual(encode_observation(base).coverage['deck_available'], 0)
        for deck in ('none', 'empty', '{}', '(0): {}'):
            base['text'] = f'Deck: {deck}\nRelics: {{}}\nPotions: none'
            e = encode_observation(base)
            self.assertEqual(e.coverage['deck_available'], 1)
            self.assertEqual(e.coverage['deck_count'], 0)
        for text in ('Deck: {Strike', 'Deck: (2): {Strike,}', 'Relics: {Pen Nib:nope,}', 'Potions: [oops] Fire Potion'):
            base['text'] = text
            with self.assertRaisesRegex(ValueError, 'Malformed|mismatch'):
                encode_observation(base)
        base['text'] = 'Deck: (1): {Mystery Card,}\nRelics: {}\nPotions: none'
        e = encode_observation(base)
        self.assertEqual(e.coverage['unk_by_namespace']['card'], 1)
        self.assertIn(UNK, e.entity_ids)
        base['text'] = 'Deck: (1): {Unknown+Name,}'
        e = encode_observation(base)
        unknown_card = (e.entity_ids == UNK) & (e.entity_types == ENTITY_TYPES['deck'])
        self.assertEqual(e.entity_numeric[unknown_card, 1][0], 0)

    def test_potion_text_explicit_empty_and_unknown_slots(self):
        obs = ooc_observation()
        # WE_MEET_AGAIN suppresses all drink/discard labels. potionCount=1
        # prints slot 0, which can be empty while an occupied slot is hidden.
        obs['state']['event'] = {'event': 'WE_MEET_AGAIN', 'potion': 'Fire Potion'}
        obs['state']['screen_state'] = 'ScreenState.EVENT_SCREEN'
        obs['legal_actions'] = [{'index': 0, 'description': 'event option 3'}]
        obs['text'] = 'Potions: EMPTY_POTION_SLOT'
        e = encode_observation(obs)
        potions = np.flatnonzero(e.entity_types == ENTITY_TYPES['potion'])
        np.testing.assert_allclose(e.entity_numeric[potions, 0], [0, 0])
        np.testing.assert_array_equal(e.entity_numeric[potions, 1], [1, 0])
        np.testing.assert_array_equal(e.entity_numeric[potions, 16], [1, 0])
        self.assertEqual(e.entity_ids[potions[0]], vocabulary_id('potion', 'empty'))
        self.assertEqual(e.entity_ids[potions[1]], PAD)  # identity unavailable
        self.assertEqual(e.coverage['potion_count'], 1)
        self.assertEqual(e.coverage['potion_slots_unresolved'], 1)
        self.assertIsNone(e.coverage['potion_capacity'])
        # The event's public potion name has no slot; do not guess its owner.
        obs['adapter_public_ooc_potions'] = []
        self.assertEqual(array_digest(e), array_digest(encode_observation(obs)))
        obs['text'] = 'Potions: Fire Potion'
        e = encode_observation(obs)
        potions = e.entity_types == ENTITY_TYPES['potion']
        self.assertEqual(e.entity_numeric[potions, 16][0], 1)
        self.assertEqual(e.coverage['potion_slots_unresolved'], 0)

    def test_ooc_native_potion_holes_raw_and_filtered_adapter(self):
        obs = ooc_observation()
        obs['text'] = 'Potions: EMPTY_POTION_SLOT'
        obs['legal_actions'] = [
            {'index': 7, 'description': 'leave shop'},
            {'index': 8, 'description': 'discard potion Fire Potion (slot 2)'}]
        raw = encode_observation(obs)
        potions = np.flatnonzero(raw.entity_types == ENTITY_TYPES['potion'])
        np.testing.assert_allclose(raw.entity_numeric[potions, 0], [0, .2])
        np.testing.assert_array_equal(raw.entity_numeric[potions, 1], [1, 0])
        self.assertEqual(raw.action_source[1], potions[1])
        self.assertEqual(raw.coverage['potion_count'], 1)
        self.assertEqual(raw.coverage['unresolved_sources'], 0)
        self.assertIsNone(raw.coverage['potion_capacity'])  # no invented slot 1
        self.assertEqual(raw.coverage['unk_by_namespace'], {})

        # Integration contract: D's shared adapter extracts from the original
        # raw public menu before hiding all OOC drink/discard choices.
        filtered = deepcopy(obs)
        filtered['legal_actions'] = obs['legal_actions'][:1]
        filtered['adapter_public_ooc_potions'] = [{'slot': 2, 'name': 'Fire Potion'}]
        encoded = encode_observation(filtered)
        for key in ('entity_ids', 'entity_types', 'entity_aux', 'entity_numeric', 'power_ids'):
            np.testing.assert_array_equal(getattr(raw, key), getattr(encoded, key))
        self.assertEqual(encoded.action_indices.tolist(), [7])
        self.assertEqual(encoded.coverage['potion_slots_unresolved'], 0)
        self.assertEqual(encoded.coverage['potion_count'], 1)
        # Only allowlisted slot/name are read, even within this adapter field.
        private = deepcopy(filtered)
        private['adapter_public_ooc_potions'][0].update(native=object(), bits=999, capacity=9, seed=124)
        private['adapter_public_potions'] = [{'slot': 1, 'name': 'Block Potion'}]
        self.assertEqual(array_digest(encoded), array_digest(encode_observation(private)))

    def test_ooc_repeated_potion_names_merge_by_slot_without_capacity(self):
        obs = ooc_observation()
        obs['text'] = 'Potions: EMPTY_POTION_SLOT, Fire Potion, EMPTY_POTION_SLOT'
        labels = ['leave shop', 'discard potion Fire Potion (slot 1)',
                  'drink potion Blood Potion (slot 3): heal 17 HP',
                  'discard potion Blood Potion (slot 3)',
                  'drink potion Blood Potion (slot 4): heal 17 HP',
                  'discard potion Blood Potion (slot 4)']
        obs['legal_actions'] = [{'index': i, 'description': s} for i, s in enumerate(labels)]
        e = encode_observation(obs)
        potions = np.flatnonzero(e.entity_types == ENTITY_TYPES['potion'])
        np.testing.assert_allclose(e.entity_numeric[potions, 0], [0, .1, .2, .3, .4])
        np.testing.assert_array_equal(e.entity_numeric[potions, 1], [1, 0, 1, 0, 0])
        self.assertEqual(e.action_source[1:].tolist(), [potions[1], potions[3], potions[3], potions[4], potions[4]])
        self.assertEqual(e.coverage['potion_count'], 3)
        self.assertEqual(e.coverage['potion_slots_unresolved'], 0)
        self.assertIsNone(e.coverage['potion_capacity'])

    def test_ooc_conflicting_potion_slots_rejected_and_adapter_is_allowlisted(self):
        obs = ooc_observation()
        obs['text'] = 'Potions: EMPTY_POTION_SLOT'
        obs['legal_actions'] = [{'index': i, 'description': s} for i, s in enumerate([
            'leave shop', 'discard potion Fire Potion (slot 2)', 'discard potion Block Potion (slot 2)'])]
        with self.assertRaisesRegex(ValueError, 'conflicting public potion slot 2'):
            encode_observation(obs)
        # An explicit adapter field takes precedence over raw-menu recovery.
        obs['adapter_public_ooc_potions'] = [{'slot': 2, 'name': 'Fire Potion'}]
        self.assertEqual(encode_observation(obs).coverage['potion_count'], 1)
        obs['legal_actions'] = obs['legal_actions'][:1]
        for records in ([{'slot': 0, 'name': 'Fire Potion'}],
                        [{'slot': 2, 'name': 'Fire Potion'}, {'slot': 2, 'name': 'Block Potion'}]):
            obs['adapter_public_ooc_potions'] = records
            with self.assertRaisesRegex(ValueError, 'conflicting public potion slot'):
                encode_observation(obs)
        for records in (None, [{'slot': -1, 'name': 'Fire Potion'}],
                        [{'slot': True, 'name': 'Fire Potion'}], [{'slot': 2, 'name': 'EMPTY_POTION_SLOT'}]):
            obs['adapter_public_ooc_potions'] = records
            with self.assertRaisesRegex(ValueError, 'Malformed'):
                encode_observation(obs)

    def test_offers_and_every_ooc_menu_family(self):
        obs = ooc_observation()
        labels = ['take card Bash+ [Attack, Basic]', 'buy card Inflame [Power, Uncommon] for 75g',
            'take relic Vajra', 'take boss relic Black Blood', 'buy relic Anchor for 150g',
            'take potion Energy Potion', 'buy potion Fire Potion for 50g', 'buy card remove for 75g',
            'event option 2: Living Wall: upgrade a card', 'take gold 25g', 'take key', 'end turn',
            'skip card reward', 'skip rewards / proceed', 'skip boss relic reward', 'card reward choice',
            'card remove', 'gain +2 max hp (Singing Bowl); lose card reward', 'leave shop',
            'rest (heal 30% of max HP)', 'smith (upgrade a card in your deck)', 'recall (obtain the Ruby Key)',
            'lift (Girya: gain 1 permanent Strength)', 'toke (Peace Pipe: remove a card from your deck)',
            'dig (Shovel: obtain a random relic)', 'skip campfire (proceed without resting)', 'open chest', 'skip chest']
        obs['legal_actions'] = [{'index': 100 + i, 'description': s} for i, s in enumerate(labels)]
        e = encode_observation(obs)
        self.assertEqual(len(e.action_ids), len(labels))
        self.assertFalse((e.action_types == ACTION_TYPES['unknown']).any())
        self.assertEqual(e.action_ids[0], vocabulary_id('card', 'Bash'))
        self.assertEqual(e.action_ids[2], vocabulary_id('relic', 'Vajra'))
        self.assertEqual(e.action_ids[6], vocabulary_id('potion', 'Fire Potion'))
        self.assertTrue((e.action_source[:7] == -1).all())
        self.assertTrue(e.action_text[8].any())
        self.assertFalse(e.action_text[:8].any())
        self.assertEqual(e.action_numeric[1, 7], np.float32(.15))
        self.assertEqual(e.coverage['parser_fallbacks'], 0)

    def test_native_library_twenty_obtain_offers_have_no_deck_sources(self):
        obs = ooc_observation()
        obs['state']['event'] = {'event': 'THE_LIBRARY'}
        # GameContext::chooseEventOption generates 20 distinct class cards;
        # describeGameAction prints an offer index, Card, and type/rarity tag.
        offers = ['Anger [Attack, Common]', 'Armaments [Skill, Common]',
                  'Carnage [Attack, Uncommon]', 'Battle Trance [Skill, Uncommon]',
                  'Bloodletting [Skill, Uncommon]', 'Bludgeon [Attack, Rare]',
                  'Body Slam [Attack, Common]', 'Cleave [Attack, Common]',
                  'Clothesline [Attack, Common]', 'Combust [Power, Uncommon]',
                  'Demon Form [Power, Rare]', 'Disarm [Skill, Uncommon]',
                  'Entrench [Skill, Uncommon]', 'Flex [Skill, Common]',
                  'Inflame [Power, Uncommon]', 'Pommel Strike [Attack, Common]',
                  'Rampage [Attack, Uncommon]', 'Shrug It Off [Skill, Common]',
                  'True Grit [Skill, Common]', 'Twin Strike [Attack, Common]']
        obs['text'] = ('Deck: (2): {Anger, Anger,}\nRelics: {}\nPotions: none\n'
                       'Choose a card to OBTAIN (add it to your deck) (chosen 0 of 1).')
        obs['legal_actions'] = [{'index': i, 'description': f'select card index {i}: {label}'}
                                for i, label in enumerate(offers)]
        e = encode_observation(obs)
        self.assertEqual(len(e.action_ids), 20)
        self.assertEqual(e.action_indices.tolist(), list(range(20)))
        self.assertTrue((e.action_source == -1).all())
        self.assertFalse(e.action_source_mask.any())
        self.assertTrue((e.action_types == ACTION_TYPES['select_card']).all())
        self.assertEqual(e.action_ids[0], vocabulary_id('card', 'Anger'))
        self.assertEqual(e.coverage['unresolved_sources'], 0)
        self.assertEqual(e.coverage['ambiguous_sources'], 0)
        self.assertEqual(e.coverage['unk_by_namespace'], {})
        # The same public card-select menu must still join owned cards for
        # actual deck operations (index is an offer index, never a deck slot).
        obs['legal_actions'] = obs['legal_actions'][:1]
        for purpose in ('REMOVE from your deck', 'UPGRADE',
                        'TRANSFORM (replace it with a random different card)',
                        'TRANSFORM and then UPGRADE', 'DUPLICATE (add another copy to your deck)',
                        'BOTTLE (it will start every combat already in your hand)',
                        'SACRIFICE in the Bonfire (it is removed from your deck)'):
            with self.subTest(purpose=purpose):
                obs['text'] = f'Deck: (2): {{Anger, Anger,}}\nChoose a card to {purpose} (chosen 0 of 1).'
                joined = encode_observation(obs)
                self.assertGreaterEqual(joined.action_source[0], 0)
                self.assertEqual(joined.entity_types[joined.action_source[0]], ENTITY_TYPES['deck'])
                self.assertEqual(joined.coverage['unresolved_sources'], 0)

    def test_public_event_aliases_and_packaged_vocabulary_resource(self):
        for public, display in (('BONFIRE_SPIRITS', 'Bonfire Elementals'), ('DESIGNER_IN_SPIRE', 'Designer')):
            with self.subTest(event=public):
                self.assertNotEqual(vocabulary_id('event', public), UNK)
                self.assertEqual(vocabulary_id('event', public), vocabulary_id('event', display))
                obs = ooc_observation()
                obs['state']['event'] = {'event': public, 'phase': 1, 'upgrade_one': True}
                encoded = encode_observation(obs)
                self.assertIn(vocabulary_id('event', public), encoded.entity_ids)
                self.assertEqual(encoded.coverage['unk_by_namespace'], {})
        # Python >=3.10 is supported; tomllib first appeared in Python 3.11.
        try:
            import tomllib
        except ImportError:
            tomllib = None
        root = Path(__file__).resolve().parents[2]
        if tomllib is not None:
            config = tomllib.loads((root / 'pyproject.toml').read_text())
            self.assertIn('entity_vocabulary.json', config['tool']['setuptools']['package-data']['sts_policy'])
            self.assertEqual(config['project']['dependencies'], [])
        from importlib.resources import files
        resource = files('sts_policy').joinpath('entity_vocabulary.json').read_bytes()
        self.assertEqual(hashlib.sha256(resource).hexdigest(), VOCABULARY_SHA256)

    def test_burning_elite_and_feasible_path_bounds(self):
        obs = ooc_observation()
        obs['state']['map'] = {'cur_y': -1, 'burning_elite': {'x': 0, 'y': 0}, 'nodes': [
            {'x': 0, 'y': 0, 'room': 'ELITE', 'burning_elite': True, 'edges': [0, 1]},
            {'x': 1, 'y': 0, 'room': 'MONSTER', 'burning_elite': False, 'edges': [1]},
            {'x': 0, 'y': 1, 'room': 'REST', 'edges': []}, {'x': 1, 'y': 1, 'room': 'ELITE', 'edges': []}]}
        obs['legal_actions'] = [{'index': 0, 'description': 'choose map node x=0 room=ELITE (burning: Emerald Key)'},
                               {'index': 1, 'description': 'choose map node x=1 room=MONSTER'}]
        e = encode_observation(obs)
        first, second = e.action_source
        self.assertEqual(e.entity_types[first], ENTITY_TYPES['map'])
        self.assertEqual(e.entity_numeric[first, 1], 1)
        self.assertEqual(e.entity_numeric[second, 1], 0)
        # Elite path min=1 max=2; branches must not be counted as both visited.
        self.assertAlmostEqual(e.entity_numeric[first, 4], 1 / 15)
        self.assertAlmostEqual(e.entity_numeric[first, 10], 2 / 15)
        obs['state']['map']['nodes'].reverse()
        self.assertEqual(array_digest(e), array_digest(encode_observation(obs)))

    def test_native_full_map_implicit_boss_edges_and_row14_transition(self):
        obs = ooc_observation()
        # Native Map::createPathsIteration adds edges [3] at row 14;
        # map_graph returns existing ordinary nodes on rows 0..14 only.
        # Six paths across the native 7-column/15-row grid, with a noncrossing
        # fork at (2,3) and merge at (4,7), then all top edges go to the boss.
        traces = [[x] * 15 for x in (0, 2, 4, 5, 6)] + [[2] * 4 + [3] * 3 + [4] * 8]
        edges = {}
        for trace in traces:
            for y, x in enumerate(trace):
                edges.setdefault((x, y), set()).add(trace[y + 1] if y < 14 else 3)
        nodes = [{'x': x, 'y': y, 'room': 'REST' if y == 14 else 'TREASURE' if y == 8 else 'MONSTER',
                  'burning_elite': False, 'edges': sorted(children)}
                 for (x, y), children in sorted(edges.items())]
        obs['state']['map'] = {'cur_y': -1, 'burning_elite': None, 'nodes': nodes}
        obs['legal_actions'] = [{'index': i, 'description': f'choose map node x={x} room=MONSTER'}
                                for i, x in enumerate((0, 2, 4, 5, 6))]
        e = encode_observation(obs)
        for source, count in zip(e.action_source, (1, 2, 1, 1, 1)):
            self.assertAlmostEqual(e.entity_numeric[source, 2], np.log1p(count) / 10)
            np.testing.assert_allclose(e.entity_numeric[source, 3:9], [13 / 15, 0, 1 / 15, 0, 0, 1 / 15])
            np.testing.assert_allclose(e.entity_numeric[source, 9:15], e.entity_numeric[source, 3:9])
        obs['state']['map']['nodes'] = list(reversed(nodes))
        self.assertEqual(array_digest(e), array_digest(encode_observation(obs)))
        obs['state']['map']['cur_y'] = 13
        obs['legal_actions'] = [{'index': 0, 'description': 'choose map node x=2 room=REST'}]
        e = encode_observation(obs)
        self.assertAlmostEqual(e.entity_numeric[e.action_source[0], 2], np.log1p(1) / 10)
        obs['state']['map']['cur_y'] = 14
        obs['legal_actions'] = [{'index': 0, 'description': 'choose map node x=0 advance to boss'}]
        e = encode_observation(obs)
        boss = e.action_source[0]
        self.assertEqual(e.entity_ids[boss], vocabulary_id('room', 'BOSS'))
        self.assertEqual(e.action_ids[0], vocabulary_id('room', 'BOSS'))
        self.assertEqual(e.entity_numeric[boss, 0], 1)
        self.assertAlmostEqual(e.entity_numeric[boss, 2], np.log1p(1) / 10)
        self.assertFalse(e.entity_numeric[boss, 3:15].any())

    def test_real_native_boss_map_raw_and_filtered_menus(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] /
                              'fixtures/native015_boss_map_case.json').read_text())
        obs = fixture['observation']  # External source/seed metadata is never policy input.
        original = deepcopy(obs)
        public_hash = hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(',', ':'),
                                               ensure_ascii=True).encode()).hexdigest()
        self.assertEqual(public_hash, fixture['public_hash'])
        self.assertEqual(public_hash, '59967f716465d45fb64f5077a4d3f6899d1f6f532447c345d77f83fd83d0b837')
        self.assertNotIn('world_seed', obs)
        self.assertNotIn('world_seed', obs['state'])
        self.assertEqual(obs['state']['map']['cur_y'], 14)
        self.assertEqual(obs['legal_actions'][0]['description'], 'choose map node x=0 advance to boss')
        self.assertTrue(all(n['y'] < 15 for n in obs['state']['map']['nodes']))
        self.assertTrue(all(n['edges'] == [3] for n in obs['state']['map']['nodes'] if n['y'] == 14))
        raw = encode_observation(obs)
        self.assertEqual(raw.action_indices.tolist(), [0, 1, 2])
        self.assertEqual(raw.action_types.tolist(), [ACTION_TYPES[k] for k in ('map', 'discard_potion', 'discard_potion')])
        self.assertTrue(raw.action_mask.all())
        self.assertTrue((raw.action_ids > UNK).all())
        boss = np.flatnonzero(raw.entity_types == ENTITY_TYPES['map'])
        self.assertEqual(boss.tolist(), raw.action_source[:1].tolist())
        self.assertEqual(raw.entity_ids[boss].tolist(), [vocabulary_id('room', 'BOSS')])
        self.assertEqual(raw.action_ids[0], vocabulary_id('room', 'BOSS'))
        self.assertEqual(raw.action_source_members[0, raw.action_source_mask[0]].tolist(), boss.tolist())
        self.assertEqual(raw.entity_numeric[boss[0], 0], 1)
        self.assertAlmostEqual(raw.entity_numeric[boss[0], 2], np.log1p(1) / 10)
        self.assertFalse(raw.entity_numeric[boss[0], 3:15].any())
        potions = np.flatnonzero(raw.entity_types == ENTITY_TYPES['potion'])
        self.assertEqual(raw.action_source[1:].tolist(), potions.tolist())
        self.assertEqual(raw.coverage['unk_by_namespace'], {})
        self.assertEqual(raw.coverage['parser_fallbacks'], 0)
        self.assertEqual(raw.coverage['unresolved_sources'], 0)
        self.assertEqual(raw.coverage['unresolved_targets'], 0)

        filtered = deepcopy(obs)
        filtered['legal_actions'] = obs['legal_actions'][:1]
        filtered['adapter_public_ooc_potions'] = []
        for action in obs['legal_actions'][1:]:
            match = re.fullmatch(r'discard potion (.+) \(slot (\d+)\)', action['description'])
            self.assertIsNotNone(match)
            filtered['adapter_public_ooc_potions'].append({'slot': int(match[2]), 'name': match[1]})
        encoded = encode_observation(filtered)
        self.assertEqual(encoded.action_indices.tolist(), [0])
        self.assertEqual(encoded.action_mask.tolist(), [True])
        self.assertEqual(encoded.coverage['action_count'], 1)
        self.assertEqual(encoded.coverage['potion_count'], 2)
        self.assertEqual(encoded.coverage['potion_slots_unresolved'], 0)
        for field in fields(EncodedEntityObservation):
            if field.name.startswith(('entity_', 'power_')):
                np.testing.assert_array_equal(getattr(raw, field.name), getattr(encoded, field.name))
            elif field.name.startswith('action_'):
                np.testing.assert_array_equal(getattr(raw, field.name)[:1], getattr(encoded, field.name))
        self.assertEqual(obs, original)

    def test_boss_action_parameter_aliases_share_context_and_preserve_indices(self):
        obs = ooc_observation()
        obs['state']['map'] = {'cur_y': 14, 'nodes': [
            {'x': 2, 'y': 14, 'room': 'REST', 'edges': [3]}]}
        obs['legal_actions'] = [{'index': 40 + i, 'description': f'choose map node x={x} advance to boss'}
                                for i, x in enumerate((0, 3, 4, 99))]
        e = encode_observation(obs)
        self.assertEqual(e.action_indices.tolist(), [40, 41, 42, 43])
        boss = np.flatnonzero(e.entity_types == ENTITY_TYPES['map'])
        self.assertEqual(len(boss), 1)
        np.testing.assert_array_equal(e.action_source, np.repeat(boss, 4))
        np.testing.assert_array_equal(e.action_ids, np.repeat(vocabulary_id('room', 'BOSS'), 4))
        np.testing.assert_array_equal(e.action_numeric, np.repeat(e.action_numeric[:1], 4, axis=0))
        np.testing.assert_array_equal(e.action_text, np.zeros_like(e.action_text))

    def test_map_missing_intermediate_or_nonboss_terminal_still_fails(self):
        obs = ooc_observation()
        obs['state']['map'] = {'cur_y': 12, 'nodes': [
            {'x': 2, 'y': 13, 'room': 'MONSTER', 'edges': [2]},
            {'x': 2, 'y': 14, 'room': 'REST', 'edges': [3]}]}
        obs['legal_actions'] = [{'index': 0, 'description': 'choose map node x=2 room=MONSTER'}]
        for row in (13, 14):
            broken = deepcopy(obs)
            broken['state']['map']['nodes'] = [n for n in broken['state']['map']['nodes'] if n['y'] != row]
            with self.subTest(missing_row=row), self.assertRaisesRegex(ValueError, 'absent|no destination'):
                encode_observation(broken)
        obs['state']['map']['nodes'][-1]['edges'] = [4]
        with self.assertRaisesRegex(ValueError, 'no destination'):
            encode_observation(obs)
        obs['state']['map']['cur_y'] = 14
        obs['legal_actions'] = [{'index': 0, 'description': 'choose map node x=4 room=BOSS'}]
        with self.assertRaisesRegex(ValueError, 'absent'):
            encode_observation(obs)

    def test_real_native_map_missing_normal_destination_still_fails(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] /
                              'fixtures/native015_boss_map_case.json').read_text())
        obs = fixture['observation']
        obs['state']['map']['cur_y'] = 13
        obs['legal_actions'][0]['description'] = 'choose map node x=0 room=REST'
        self.assertEqual(encode_observation(obs).action_indices.tolist(), [0, 1, 2])
        obs['state']['map']['nodes'] = [n for n in obs['state']['map']['nodes']
                                       if (n['x'], n['y']) != (0, 14)]
        with self.assertRaisesRegex(ValueError, r'legal map destination \(0, 14\) absent from public graph'):
            encode_observation(obs)

    def test_recent_known_play_and_end_turn_are_parsed(self):
        obs = combat_observation()
        obs['state']['combat']['recent_actions'] = [
            'play Bash [Attack] (cost 2) -> SENTRY [enemy 0] (deal 8)', 'end turn']
        encoded = encode_observation(obs)
        self.assertEqual(encoded.coverage['recent_total'], 2)
        self.assertEqual(encoded.coverage['recent_unparsed'], 0)
        self.assertEqual(encoded.recent_types.tolist(), [ACTION_TYPES['play'], ACTION_TYPES['end_turn']])
        self.assertEqual(encoded.recent_ids.tolist(), [vocabulary_id('card', 'Bash'),
                                                     vocabulary_id('action', 'end_turn')])
        self.assertEqual(encoded.recent_numeric[0, 3], np.float32(2 / 5))
        self.assertTrue(encoded.recent_numeric[0, 19])
        self.assertFalse(encoded.recent_numeric[1].any())
        self.assertEqual(encode_observation(ooc_observation()).coverage['recent_unparsed'], 0)

    def test_recent_unparsed_counts_unknown_entries_without_changing_history(self):
        obs = combat_observation()
        selection = 'select card for HEADBUTT: Strike [Attack] (cost 1)'
        obs['state']['combat']['recent_actions'] = [
            'end turn', selection, 'keep current hand (discard no cards)',
            'drink potion Fire Potion -> SENTRY [enemy 1]', 'exhaust no cards',
            'discard potion Block Potion', selection,
            'exhaust: hand [0] Strike [Attack] (cost 1)']
        kinds = ['end_turn', 'unknown', 'unknown', 'drink_potion', 'unknown',
                 'discard_potion', 'unknown', 'select_many']
        encoded = encode_observation(obs)
        self.assertEqual(encoded.coverage['recent_total'], len(kinds))
        self.assertEqual(encoded.coverage['recent_unparsed'], 4)  # Entries, including repeats.
        self.assertEqual(encoded.coverage['parser_fallbacks'], 0)  # Current menu parses fully.
        self.assertEqual(encoded.coverage['unk_by_namespace'], {})  # action:unknown is a known token.
        self.assertEqual(encoded.recent_types.tolist(), [ACTION_TYPES[k] for k in kinds])
        self.assertEqual(encoded.recent_ids.tolist(), [vocabulary_id('action', k) for k in kinds])
        self.assertTrue(encoded.recent_mask.all())
        self.assertFalse(encoded.recent_numeric.any())
        # Current-menu fallbacks are counted independently of recent history.
        obs['legal_actions'].append({'index': 18, 'description': 'unrecognized public choice'})
        fallback = encode_observation(obs)
        self.assertEqual(fallback.coverage['parser_fallbacks'], 1)
        self.assertEqual(fallback.coverage['recent_unparsed'], 4)
        for name in ('recent_ids', 'recent_types', 'recent_numeric', 'recent_mask'):
            np.testing.assert_array_equal(getattr(encoded, name), getattr(fallback, name))

    def test_recent_order_no_cap_and_no_entity_or_action_truncation(self):
        obs = combat_observation()
        a = encode_observation(obs)
        obs['state']['combat']['recent_actions'].reverse()
        b = encode_observation(obs)
        np.testing.assert_array_equal(a.recent_ids[::-1], b.recent_ids)
        np.testing.assert_array_equal(a.entity_ids, b.entity_ids)
        obs['state']['combat']['recent_actions'] *= 40
        obs['state']['combat']['cards']['hand'] = [card(index=i) for i in range(12)]
        obs['legal_actions'] = [{'index': i, 'description': 'end turn'} for i in range(140)]
        e = encode_observation(obs)
        self.assertEqual(len(e.recent_ids), 80)
        self.assertEqual(e.coverage['recent_truncated'], 0)
        self.assertEqual(sum(e.entity_types == ENTITY_TYPES['hand']), 12)
        self.assertEqual(len(e.action_ids), 140)

    def test_collation_masks_padding_and_empty_menu(self):
        a, b = combat_observation(), ooc_observation()
        b['legal_actions'] = b['legal_actions'][:1]
        ea, eb = encode_observation(a), encode_observation(b)
        batch = collate_observations([ea, eb])
        self.assertEqual(batch['action_mask'].sum(1).tolist(), [8, 1])
        self.assertEqual(batch['recent_mask'].sum(1).tolist(), [2, 0])
        self.assertTrue((batch['action_source'][1, 1:] == -1).all())
        self.assertTrue((batch['action_ids'][1, 1:] == PAD).all())
        self.assertEqual(batch['entity_counts'].tolist(), [len(ea.entity_ids), len(eb.entity_ids)])
        a['legal_actions'] = []
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            encode_observation(a)
        with self.assertRaises(ValueError):
            collate_observations([])

    def test_fresh_process_determinism_and_vocab_regeneration(self):
        code = ('from tests.unit.test_entity_encoding import combat_observation,array_digest;'
                'from sts_policy.entity_encoding import ENCODER_SHA256,encode_observation;'
                'print(ENCODER_SHA256,array_digest(encode_observation(combat_observation())))')
        outputs = [subprocess.check_output([sys.executable, '-c', code], env={**os.environ, 'PYTHONHASHSEED': seed}, text=True).strip()
                   for seed in ('1', '9817')]
        self.assertEqual(outputs[0], outputs[1])
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'vocabulary.json'
            subprocess.run([sys.executable, str(root / 'scripts/rl/build_entity_vocabulary.py'), '--output', str(out)],
                           check=True, capture_output=True)
            self.assertEqual(hashlib.sha256(out.read_bytes()).hexdigest(), VOCABULARY_SHA256)
        self.assertEqual(vocabulary_id('card', 'STRIKE'), vocabulary_id('card', 'strike'))
        self.assertEqual(vocabulary_id('enemy', 'ACID_SLIME_L'), vocabulary_id('enemy', 'Acid Slime (L)'))
        self.assertNotEqual(vocabulary_id('card', 'J.A.X.'), UNK)
        self.assertEqual(vocabulary_id('card', 'Searing_Blow+2'), vocabulary_id('card', 'Searing Blow'))


if __name__ == '__main__':
    unittest.main()
