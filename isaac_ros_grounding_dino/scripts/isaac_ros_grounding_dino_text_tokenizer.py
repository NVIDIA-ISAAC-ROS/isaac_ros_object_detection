#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from isaac_ros_grounding_dino_interfaces.srv import GetTextTokens
from isaac_ros_tensor_list_interfaces.msg import Tensor
import numpy as np
import rclpy
from rclpy.node import Node
from transformers import AutoTokenizer

MAX_TEXT_LENGTH = 256


class TensorType(Enum):
    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    INT64 = 7
    UINT64 = 8
    FLOAT32 = 9
    FLOAT64 = 10


class GroundingDinoTextTokenizer(Node):

    def __init__(self):
        super().__init__('grounding_dino_text_tokenizer')

        self.text_tokenizer_service = self.create_service(
            GetTextTokens, 'get_text_tokens', self.get_text_tokens_callback)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased', local_files_only=True)
        except OSError:
            self.get_logger().warn(
                'Tokenizer not found in local cache. Fetching model from the Hugging Face Hub.')
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            except Exception as e:
                self.get_logger().error(
                    f'Failed to fetch tokenizer from Hugging Face Hub: {e}')
                self.get_logger().error(
                    'Please check your internet connection or manually download the model.')
                raise RuntimeError(
                    'Could not load tokenizer from local cache or download from internet')

        self.special_tokens = ['[CLS]', '[SEP]', '.', '?']
        self.special_token_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens)

    def generate_text_token_mask_and_pos_ids(self, tokenized):
        """Generate self-attention mask and position ids between each pair of special tokens."""
        input_ids = tokenized['input_ids'].squeeze(0)
        num_token = len(input_ids)

        # Find special token positions
        special_positions = np.where(np.isin(input_ids, self.special_token_ids))[0]

        # Initialize outputs
        text_token_mask = np.eye(num_token, dtype=np.uint8)
        position_ids = np.zeros(num_token, dtype=np.int64)

        # Process special tokens in order
        prev_pos = 0
        for col in special_positions:
            if col not in (0, num_token - 1):
                # Set self-attention mask for segment
                text_token_mask[prev_pos + 1:col + 1, prev_pos + 1:col + 1] = 1
                # Set position ids for segment
                position_ids[prev_pos + 1:col + 1] = np.arange(col - prev_pos)
                prev_pos = col

        return text_token_mask, position_ids

    def create_pos_map(self, tokenized, class_ids, text):
        """Construct a map where pos_map[i,j] = 1 iff category i is associated to token j."""
        pos_map = np.zeros((len(class_ids), MAX_TEXT_LENGTH), dtype=np.uint8)

        for i, category in enumerate(class_ids):
            start_char = text.find(category)
            end_char = start_char + len(category) - 1
            start_token = tokenized.char_to_token(start_char)
            end_token = tokenized.char_to_token(end_char)

            # Try fallback positions if needed
            if end_token is None and end_char > 0:
                end_token = tokenized.char_to_token(end_char - 1)
            if end_token is None and end_char > 1:
                end_token = tokenized.char_to_token(end_char - 2)

            # Fill positive map if valid positions found
            if start_token is not None and end_token is not None and start_token <= end_token:
                pos_map[i, start_token:end_token + 1] = 1

        return pos_map

    def tokenize_text(self, text):
        """Generate text token tensors and positive map."""
        tokenized = self.tokenizer(text, padding='max_length', return_tensors='np',
                                   max_length=MAX_TEXT_LENGTH)

        # Parse categories from text. Remove empty strings and strip whitespace.
        class_ids = [cat.strip() for cat in text.split('.') if cat.strip()]

        # Create positive map
        pos_map = self.create_pos_map(tokenized, class_ids, text)

        # Generate text token mask and position ids
        text_token_mask, position_ids = self.generate_text_token_mask_and_pos_ids(tokenized)

        # Add batch dimension back and truncate
        text_token_mask = text_token_mask[np.newaxis, :MAX_TEXT_LENGTH, :MAX_TEXT_LENGTH]
        position_ids = position_ids[np.newaxis, :MAX_TEXT_LENGTH]

        return (
            (
                tokenized['input_ids'][:, :MAX_TEXT_LENGTH].astype(np.int64),
                tokenized['attention_mask'][:, :MAX_TEXT_LENGTH].astype(np.uint8),
                tokenized['token_type_ids'][:, :MAX_TEXT_LENGTH].astype(np.int64),
                position_ids,
                text_token_mask,
            ),
            pos_map,
            class_ids,
        )

    def create_tensor_msg(self, data, name, data_type):
        """Create a Tensor message from numpy tensor data."""
        tensor_msg = Tensor()
        tensor_msg.name = name
        tensor_msg.data_type = data_type
        tensor_msg.data = data.tobytes()
        tensor_msg.shape.rank = len(data.shape)
        tensor_msg.shape.dims = list(data.shape)
        return tensor_msg

    def get_text_tokens_callback(self, request, response):
        """Service callback to tokenize text and return tensors."""
        text_tensors, pos_maps, class_ids = self.tokenize_text(request.prompt)

        tensor_names = ['input_ids', 'attention_mask', 'token_type_ids',
                        'position_ids', 'text_token_mask']
        tensor_types = [TensorType.INT64, TensorType.UINT8, TensorType.INT64,
                        TensorType.INT64, TensorType.UINT8]
        response.text_tensors.tensors = [
            self.create_tensor_msg(data, name, dtype.value)
            for data, name, dtype in zip(text_tensors, tensor_names, tensor_types)]

        response.pos_maps = self.create_tensor_msg(
            pos_maps, 'pos_maps', TensorType.UINT8.value)

        response.class_ids = class_ids

        return response


def main():
    rclpy.init()
    rclpy.spin(GroundingDinoTextTokenizer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
