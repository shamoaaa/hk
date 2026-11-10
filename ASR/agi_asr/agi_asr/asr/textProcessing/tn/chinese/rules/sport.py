# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from asr.textProcessing.tn.chinese.rules.cardinal import Cardinal
from asr.textProcessing.tn.processor import Processor

from pynini import cross, string_file
from pynini.lib.pynutil import delete, insert

import os
from pathlib import Path
WEIGHTS = Path(os.getenv("MS_WEIGHTS", "/usr/local/hikvision/Algo_server/weights"))

class Sport(Processor):

    def __init__(self):
        super().__init__(name='sport')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        country = string_file(str(WEIGHTS/'textProcessing/tn/chinese/data/sport/country.tsv'))
        club = string_file(str(WEIGHTS/'textProcessing/tn/chinese/data/sport/club.tsv'))
        rmsign = delete('/') | delete('-') | delete(':')
        rmspace = delete(' ').ques

        number = Cardinal().number
        score = rmspace + number + rmsign + insert('比') + number + rmspace
        tagger = (insert('team: "') + (country | club) + insert('" score: "') +
                  score + insert('"'))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        super().build_verbalizer()
        team = delete('team: "') + self.SIGMA + delete('" ')
        score = delete('score: "') + self.SIGMA + delete('"')
        verbalizer = team + score
        self.verbalizer |= self.delete_tokens(verbalizer)
