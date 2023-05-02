/*
 * aidadsp-lv2
 * Copyright (C) 2022-2023 Massimo Pennazio <maxipenna@libero.it>
 * Copyright (C) 2023 Filipe Coelho <falktx@falktx.com>
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "rt-neural-generic.h"
#include "models.cpp"
#include <strstream>

using namespace models;

static const struct {
    const char* const data;
    const unsigned int size;
} kModels[] = {
    { AMP_Blues_Deluxe_Clean1Data, AMP_Blues_Deluxe_Clean1DataSize },
    { AMP_Blues_Deluxe_Clean2Data, AMP_Blues_Deluxe_Clean2DataSize },
    { AMP_Blues_Deluxe_Clean3Data, AMP_Blues_Deluxe_Clean3DataSize },
    { AMP_Blues_Deluxe_CrunchyData, AMP_Blues_Deluxe_CrunchyDataSize },
    { AMP_Blues_Deluxe_DirtyData, AMP_Blues_Deluxe_DirtyDataSize },
    { AMP_Blues_Deluxe_GainyData, AMP_Blues_Deluxe_GainyDataSize },
    { AMP_Marsh_JVM_Clean1Data, AMP_Marsh_JVM_Clean1DataSize },
    { AMP_Marsh_JVM_Clean2Data, AMP_Marsh_JVM_Clean2DataSize },
    { AMP_Marsh_JVM_CrunchData, AMP_Marsh_JVM_CrunchDataSize },
    { AMP_Marsh_JVM_OD1Data, AMP_Marsh_JVM_OD1DataSize },
    { AMP_Marsh_JVM_OD2Data, AMP_Marsh_JVM_OD2DataSize },
    { AMP_Orange_CleanData, AMP_Orange_CleanDataSize },
    { AMP_Orange_Crunchy1Data, AMP_Orange_Crunchy1DataSize },
    { AMP_Orange_Crunchy2Data, AMP_Orange_Crunchy2DataSize },
    { AMP_Orange_DirtyData, AMP_Orange_DirtyDataSize },
    { AMP_Orange_NastyData, AMP_Orange_NastyDataSize },
    { AMP_Twin_Custom1Data, AMP_Twin_Custom1DataSize },
    { AMP_Twin_Custom2Data, AMP_Twin_Custom2DataSize },
    { AMP_Twin_Vintage1Data, AMP_Twin_Vintage1DataSize },
    { AMP_Twin_Vintage2Data, AMP_Twin_Vintage2DataSize },
};

DynamicModel* RtNeuralGeneric::loadModelFromIndex(LV2_Log_Logger* logger, int modelIndex, int* input_size_ptr)
{
    static_assert(sizeof(kModels)/sizeof(kModels[0]) == 20, "expected number of models");
    if (modelIndex == 0 || modelIndex > sizeof(kModels)/sizeof(kModels[0]))
        return nullptr;

    int input_skip;
    int input_size;
    float input_gain;
    float output_gain;
    float model_samplerate;
    nlohmann::json model_json;

    try {
        std::istrstream jsonStream(kModels[modelIndex - 1].data, kModels[modelIndex - 1].size);
        jsonStream >> model_json;

        /* Understand which model type to load */
        input_size = model_json["in_shape"].back().get<int>();
        if (input_size > MAX_INPUT_SIZE) {
            throw std::invalid_argument("Value for input_size not supported");
        }

        if (model_json["in_skip"].is_number()) {
            input_skip = model_json["in_skip"].get<int>();
            if (input_skip > 1)
                throw std::invalid_argument("Values for in_skip > 1 are not supported");
        }
        else {
            input_skip = 0;
        }

        if (model_json["in_gain"].is_number()) {
            input_gain = DB_CO(model_json["in_gain"].get<float>());
        }
        else {
            input_gain = 1.0f;
        }

        if (model_json["out_gain"].is_number()) {
            output_gain = DB_CO(model_json["out_gain"].get<float>());
        }
        else {
            output_gain = 1.0f;
        }

        if (model_json["metadata"]["samplerate"].is_number()) {
            model_samplerate = model_json["metadata"]["samplerate"].get<float>();
        }
        else if (model_json["samplerate"].is_number()) {
            model_samplerate = model_json["samplerate"].get<float>();
        }
        else {
            model_samplerate = 48000.0f;
        }

        lv2_log_note(logger, "Successfully loaded json file\n");
    }
    catch (const std::exception& e) {
        lv2_log_error(logger, "Unable to load json file, error: %s\n", e.what());
        return nullptr;
    }

    std::unique_ptr<DynamicModel> model = std::make_unique<DynamicModel>();

    try {
        if (! custom_model_creator (model_json, model->variant))
            throw std::runtime_error ("Unable to identify a known model architecture!");

        std::visit (
            [&model_json] (auto&& custom_model)
            {
                using ModelType = std::decay_t<decltype (custom_model)>;
                if constexpr (! std::is_same_v<ModelType, NullModel>)
                {
                    custom_model.parseJson (model_json, true);
                    custom_model.reset();
                }
            },
            model->variant);
    }
    catch (const std::exception& e) {
        lv2_log_error(logger, "Error loading model: %s\n", e.what());
        return nullptr;
    }

    /* Save extra info */
    model->input_skip = input_skip != 0;
    model->input_gain = input_gain;
    model->output_gain = output_gain;
    model->samplerate = model_samplerate;
    model->param1Coeff.setSampleRate(model_samplerate);
    model->param1Coeff.setTimeConstant(0.1f);
    model->param1Coeff.setTargetValue(0.f);
    model->param1Coeff.clearToTargetValue();
    model->param2Coeff.setSampleRate(model_samplerate);
    model->param2Coeff.setTimeConstant(0.1f);
    model->param2Coeff.setTargetValue(0.f);
    model->param2Coeff.clearToTargetValue();

    /* pre-buffer to avoid "clicks" during initialization */

    {
        float out[2048] = {};
        applyModel(model.get(), out, 2048);
    }

    // cache input size for later
    *input_size_ptr = input_size;

    return model.release();
}
