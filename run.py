from par_config import *
import random
def run():
    version = st.selectbox("Model Version", list(VERSION2SPECS.keys()), 0)
    version_dict = VERSION2SPECS[version]
    mode = "txt2img"
    set_lowvram_mode(st.checkbox("Low vram mode", True))

    if version.startswith("SDXL-base"):
        add_pipeline = st.checkbox("Load SDXL-refiner?", False)
    else:
        add_pipeline = False

    seed = st.sidebar.number_input("seed", value=random.randint(1, 100), min_value=0, max_value=int(1e9))
    seed_everything(seed)

    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    is_legacy = version_dict["is_legacy"]

    prompt = input("Enter prompt: ")
    negative_prompt = ""
    stage2strength = None
    finish_denoising = False

    if add_pipeline:
        version2 = st.selectbox("Refiner:", ["SDXL-refiner-1.0", "SDXL-refiner-0.9"])
        version_dict2 = VERSION2SPECS[version2]
        state2 = init_st(version_dict2, load_filter=False)
        st.info(state2["msg"])

        stage2strength = st.number_input(
            "**Refinement strength**", value=0.15, min_value=0.0, max_value=1.0
        )

        sampler2, *_ = init_sampling(
            key=2,
            img2img_strength=stage2strength,
            specify_num_samples=False,
        )
        finish_denoising = st.checkbox("Finish denoising with refiner.", True)
        if not finish_denoising:
            stage2strength = None
    out = run_txt2img(
        prompt,
        negative_prompt,
        state,
        version,
        version_dict,
        is_legacy=is_legacy,
        return_latents=add_pipeline,
        filter=state.get("filter"),
        stage2strength=stage2strength, )

    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None

    if add_pipeline and samples_z is not None:
        samples = apply_refiner(
            samples_z,
            state2,
            sampler2,
            samples_z.shape[0],
            prompt=prompt,
            negative_prompt=negative_prompt if is_legacy else "",
            filter=state.get("filter"),
            finish_denoising=finish_denoising,
        )

    perform_save_locally("outputs/demo/txt2img/SDXL-base-1.0/samples", samples)
    print("Done !")

if __name__ == "__main__":
    run()
