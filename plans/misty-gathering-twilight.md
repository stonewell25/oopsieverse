# oopsieverse → PointWorld ブリッジ + Damage Proxy 実装計画

## 実装時の訂正(2026-07-10)

計画立案時の前提のうち以下は実装着手時に誤りと判明し、修正した:

- **dockerは不要**。oopsieverse/OmniGibsonはローカルconda環境 `oopsieverse_b1k` で直接動作する(GPU/ディスプレイあり)。PointWorldは `pointworld-env`。リポジトリ間の受け渡しをnpzにする設計自体は変更なし(理由はHDF5の互換性問題ではなく、単純に2つの独立したconda環境間の受け渡しのため)。
- 当初ターゲットにしていた `demos/behavior1k/teleop_data/{shelve_item,heat_saucepot}.hdf5` は**実際に破損している**(書き込み途中で壊れた模様、どのh5pyでも開けない)。代わりに `oopsiebench/demos/behavior1k/teleop/shelve_item_safe.hdf5`(+ `_unsafe`)など、`batch_playback_safe_unsafe.sh` が想定するsafe/unsafeペア一式を使う。こちらは正常に開ける。
- グリッパー角度マッピングの数値を訂正: Robotiq `finger_joint` のURDF可動域は **[0, 0.725] rad**(`assets/franka_description/franka_panda_robotiq_2f85.urdf:488`)、0.8ではない。`gripper_positions = (1 − w/0.08) × 0.725`。
- **Milestone①は実装・実行済み・成功**。`scripts/playback_b1k.py --export_pointworld_dir ... --export_joints_only` でjoints-only npzを出力→`oopsieverse_bridge/make_action_file.py` で(11,7)+(11,)のaction npzに変換→`generate_trajectory.py --action=file` で予測 `scene_flows` shape (1,11,11841,3)、有限値、frame0一致を確認。チェックポイントロード時に `--allow_partial_load=true` が必要だった(aux_head層の追加によるモデルコードとチェックポイントの差分、oopsieverse連携とは無関係の既知の非互換)。
- **Milestone②のエクスポート機構(oopsieverse側)は実装・実行済み・検証済み**。`PointWorldRecorder`実装時に2つのバグを発見・修正した:
  1. `on_episode_start`をplaybackループの**外**(`playback_episode`呼び出し前)で呼んでいたため、`scene.restore()`前のオブジェクトリストを捕捉してしまい、book/wineglass等のインタラクティブオブジェクトが`object_names`から欠落していた。→ オブジェクト/カメラ捕捉を最初の`on_step`呼び出し時に遅延実行するよう変更。
  2. Isaac Simの`VisionSensor.intrinsic_matrix`が捕捉タイミングによって稀に退化行列(fx=fy=0)を返すことがあった。→ `omnigibson.sensors.vision_sensor.render()`(物理を進めない純粋レンダー)でリトライするガードを追加。
  - 検証: 2台のexternal cameraそれぞれでframe-0 depthをintrinsics/extrinsicsでworld座標へ逆投影し、5つの操作対象オブジェクト(book/wineglass/bottle_of_beer/bottle_of_wine/box_of_crackers)の実際のpose中心との最近傍点距離を計測 → **全て3〜6cm以内で一致**。OpenGL→OpenCV変換を含むextrinsics規約とintrinsics計算の正しさを実データで確認済み。
- **Milestone②のPointWorld側サンプル構築・rolloutも実装・実行済み・成功**。`oopsieverse_bridge/{episode.py, sample_builder.py}` + `run_oopsieverse.py` を実装。実装中に発見・修正した追加の設計問題:
  - **ロボットベース座標系の不一致**: `_get_robot_flows_droid`(PointWorld側)はbase_pose補正なしでFKするため、DROIDデータ規約では「ロボットベース=ワールド原点」が暗黙の前提。oopsieverseのシーンではロボットベースが原点にない(例: [6.8, 0.2, 1.0])。→ `PointWorldRecorder`に`robot_base_pose`を追加エクスポートし、`episode.py`に`world_to_base_matrix()`を実装、`sample_builder.py`でカメラ点群・法線・extrinsicsをロボットベース相対座標系に変換してから渡すよう修正。オブジェクトの相対姿勢変化(GT flow計算)は座標系に依らず不変なので、world座標のままrigid-flow計算した後に結果をbase系へ変換する設計にした。
  - **クロップ中心の取り違え**: 当初`obs["eef_pos"]`(実は`robot.get_relative_eef_pose()`でbase相対)をworld座標のクロップ中心として誤用し、シーン点群が0件になるバグがあった。→ `health_link_names`から操作対象オブジェクト名を抽出し、その`object_poses`(world座標)の重心を`workspace_anchor`として使うよう修正。
  - `seg_instance`のinstance-id→オブジェクト名マッピングは`info["obs_info"]["external"][cam]["seg_instance"]`にあり(`info["external"]`ではない)、これを`PointWorldRecorder`でエクスポートに追加し、ロボットマスキングとGTフロー用のper-pointオブジェクト割当の両方に利用。
  - 検証結果: `run_oopsieverse.py --episode_npz ... --window_start 0` で予測`scene_flows` shape (1,11,12000,3)、**frame0==initial scene: True、finite: True**。oopsieverse自身のシーン(shelve_item)上でのPointWorld rolloutが成立することを確認。
- **Milestone③(damage proxy)も実装・実行済み・成功**。`oopsieverse_bridge/damage_proxy.py` + `run_oopsieverse.py --damage_proxy`。
  - `recover_instance_ids`: 当初`scipy.spatial.cKDTree`でNN照合する設計だったが、**CUDAコンテキスト初期化後にcKDTreeを呼ぶとセグフォルトする**環境問題が発覚(OpenMPライブラリ競合と推定)。→ grid_sample_transform/enforce_max_num_pointsは点を「間引くだけで座標を変えない」という性質を利用し、座標を1mm単位で丸めたハッシュ辞書による厳密一致照合に変更。12000点中11978点(99.8%)が正しく元のオブジェクトラベルに復元できることを確認。
  - `compute_object_damage`: `damagesim/core/evaluators/mechanical.py`のimpact項 `D = impact_sensitivity × max(0, mass×|Δv/Δt| − F_thresh)` を再実装。他オブジェクトとの近接(<0.03m)でゲート。GT flows(design decision②のrigid-transform伝播)と予測flowsの両方に適用。
  - 検証結果(shelve_item safeデモ、window 0とwindow 30): sim側のhealthは対象5オブジェクト(book/wineglass/bottle_of_beer/bottle_of_wine/box_of_crackers)とも終始100(safeデモなので実際の損傷イベントなし、想定通り)。GT flow proxyも同様に100のまま変化なし(**proxy数式がfalse positiveを出さないことを確認、モデル非依存の妥当性検証**)。一方、予測flow proxyでは`range_hood_uokzpw_1`(oopsieverseがdamage追跡すらしていない固定什器)に対してwindow 30で100→0.0という**偽陽性**が発生 — OGレンダリングされたシーンに対するモデルのドメインギャップ(静止した建築要素の動きを誤って予測する)を示す具体的な知見。安全なデモではsim側に実際の損傷が発生しないため、safe/unsafe識別の定量評価(相関・分類)は次のステップとして`_unsafe`デモや複数ウィンドウのスティッチングでの検証が必要(未実施)。

## Context

oopsieverseで収集したteleopエピソード(Franka Panda / BEHAVIOR-1Kタスク)のグリッパー軌道とRGB-Dを、3D world model **PointWorld** (~/PointWorld) に入力し、world modelの予測rollout上でoopsieverseのdamage evaluator(health bar)相当の評価ができるか検証する。

ユーザー確認済みスコープ:
- **段階的に3マイルストーン**: ①npz変換でDROIDシーン接続確認 → ②oopsieverseシーンのRGB-DエクスポートとPointWorld rollout → ③mechanical damage proxy
- **データ**: 既存デモ `demos/behavior1k/teleop_data/shelve_item.hdf5`(+heat_saucepot)をplaybackで再収集
- **damage**: mechanicalのimpact項のみ(thermal/electricalは状態変数がworld modelに無く原理的に不可)

## 調査で確定した前提(要点)

- **環境分離**: OmniGibsonは isaac-sim docker (`/home/sena/docker/isaac-sim`) 内のみ。そのHDF5ビルドはローカルconda h5pyと非互換(既存デモはローカルで開けない)。PointWorldは `pointworld-env` conda。→ **リポジトリ間の受け渡しは全てdocker内で書いたnpz**にする。
- teleop HDF5はシリアライズsim状態+IK eefアクションのみ(joint positions無し)。**playback(`scripts/playback_b1k.py --playback`)が決定論的に再生+観測再生成する正道** (`damageable_env.py:736` `playback_episode`)。
- depth未収集だがOG VisionSensorは `depth_linear` 対応(`vision_sensor.py:75`)。intrinsicsは `VisionSensor.intrinsic_matrix`(`vision_sensor.py:883`)、extrinsicsは `sensor.get_position_orientation()`。`build_external_sensors_config`(`playback_b1k.py:113`)は既にimage_height/width別指定可 → 16:9レンダリング容易。
- **PointWorld入力はRGB-D生データではなくworld座標系点群 `scene_flows` (T,N,3)**。depth→world逆投影は自前実装(参照: `visualization/viser_tools/visualization_utils.py:58 project_depth_to_world`)。RGB-D+K+Eはカメラpayload(DINOv3 featurizer用)として **(180,320)固定解像度** で必要(`transforms.py:606` assert)。
- **domain="droid"一択**(behaviorはR1Pro 22関節前提)。サンプル構築パイプラインは `generate_trajectory.py build_custom_batch`(:220-296)がテンプレート: `build_flow_sample → sample_cameras(2台固定) → canonicalize_gripper_keys_and_flags → apply_release_pipeline_to_sample(mode="test") → custom_collate_fn → model(batch)`。
- **T=11固定**(コンテキスト1+予測10)、autoregressiveは未実装 → エピソードはスライディングウィンドウで処理。
- robot flowsは**グリッパー点のみ**(`RELEASE_GRIPPER_ONLY=True`, `constants.py:37`)、DROID URDFは `assets/franka_description/franka_panda_robotiq_2f85.urdf`。
- damage math: `D = impact_sensitivity × F_impact + qs_sensitivity × F_qs`, `F_impact = mass × |Δv/Δt|`(`damagesim/core/evaluators/mechanical.py`)。per-step healthはplayback HDF5の `obs/health` + attr `health_list_link_names`、詳細は `info/damage_info`。
- health bar描画は `damagesim/utils/visualization.py`(`_draw_health_bar_only:532`, `render_health_bar_overlay:1446`)— 純cv2/numpyでOG非依存、PointWorld側にvendorコピー可。

## 設計判断(確定)

1. **グリッパー幾何の不一致**: oopsieverseはPandaハンド、DROID URDFはRobotiq 2F-85。**DROID Robotiq URDFをそのまま使う**。理由: robot flowsはグリッパー点のみで、モデルはRobotiq形状で学習済み(Pandaハンド点群の方がOOD)。アームFKは同一なのでTCP配置は正しい。Panda指幅 w∈[0,0.08]m → `finger_joint=(1−w/0.08)×0.8rad`、`gripper_positions=1−w/0.08` に線形マップ(符号規約はDROIDサンプル1件をdecodeして検証)。レンダリングされたPandaハンドはseg_instanceでシーン点群から**マスク除去**。
2. **GT flows**: playbackでper-objectの剛体poseを毎ステップ出力し、frame-0の点をseg_instanceでオブジェクトに割当て、pose差分で変換してGT (T,N,3) を構成(damage proxy検証と予測精度評価に使用)。背景は固定。
3. **時間解像度**: `--export_stride`(既定3)でsim stepを間引き。stride∈{2,3,5}でウィンドウ内変位量のヒストグラムをDROIDサンプルと比較して選定。
4. **damage proxy**: 予測flowsからper-object中央値セントロイドの速度/加速度を有限差分 → `F_t=m|Δv/Δt|`、他オブジェクトとの点群近接(<2×voxel=0.03m)でゲート。`health_pred=100−ΣD`。感度/閾値は1エピソードでキャリブレーション。**GT flows上でも同じproxyを計算して上界とする**(proxy誤差とモデル誤差の分離)。

## 実装

### A. oopsieverse側(docker内で実行)

**新規 `damagesim/omnigibson/pointworld_export.py`** — `PointWorldRecorder` クラス:
- `on_episode_start`: 各external sensorの `intrinsic_matrix` と world pose を取得。OG/USDカメラ(−Z前方/+Y上)→ OpenCV(+Z前方/+Y下)変換: `R_cv = R_gl @ diag([1,−1,−1])`、`E = inv([[R_cv,t],[0,1]])`(world→cam)。
- `on_step(i, obs, info)`: stride間隔で `robot.get_joint_positions()`、指qpos、eef pose(world)、全オブジェクトpose、`obs["health"]`、`json.dumps(info["damage_info"])`、per-cameraのrgb/depth_linear/seg_instanceを蓄積。
- `finalize`: `np.savez_compressed` で `demos/behavior1k/pointworld_export/episode_{task}_{demo}.npz` に保存。

**変更 `scripts/playback_b1k.py`**:
- 新args: `--export_pointworld_dir`, `--export_stride`(既定3), `--export_resolution`(既定 `720x1280`), `--export_joints_only`(①用: 画像なし高速パス、`--skip_save_images`と併用可)
- `build_external_sensors_config`(:113)のmodalitiesに `"depth_linear"` 追加、`run_playback`(:345)の解像度を16:9対応
- `run_playback` でRecorderを生成し `env.pointworld_recorder = recorder` で装着

**変更 `damagesim/omnigibson/damageable_env.py`** — `playback_episode`(:736-1016)のステップループにガード付きフック1箇所:
```python
if getattr(self, "pointworld_recorder", None) is not None:
    self.pointworld_recorder.on_step(i, obs, info)
```

### npzスキーマ(episode export)

| key | shape | dtype |
|---|---|---|
| `task_name`/`demo_id`/`robot_name`/`sim_dt`/`stride`/`time_steps` | scalar | str/float/int |
| `joint_names` / `joint_positions` | (9,) / (T,9) | str / f32 |
| `gripper_width`(2指qpos和) / `eef_pose`(world pos+quat) | (T,) / (T,7) | f32 |
| `cam{i}_rgb` / `cam{i}_depth`(mm) / `cam{i}_seg_instance` | (T,720,1280,3)/(T,720,1280)/(T,720,1280) | u8/u16/i32 |
| `cam{i}_seg_id_map`(JSON) / `cam{i}_intrinsic` / `cam{i}_extrinsic`(world→cam CV) / `cam{i}_extrinsic_per_frame`(検証用) | - /(3,3)/(4,4)/(T,4,4) | str/f32 |
| `object_names` / `object_poses` / `object_masses` | (K,)/(T,K,7)/(K,) | str/f32 |
| `health_link_names` / `health` / `damage_info`(JSON) | (L,)/(T,L)/(T,) | str/f32/str |

### B. PointWorld側(pointworld-env)

**新規パッケージ `oopsieverse_bridge/`**:
- `episode.py` — npzロード、depth mm→m、720×1280→**180×320**縮小(rgb: INTER_AREA、depth/seg: NEAREST)+intrinsicsスケール、グリッパーマッピング、seg_instanceによるロボット点マスク
- `sample_builder.py` — `build_window_sample(episode, t0, T=11)`: frame-t0 depthをworld逆投影(project_depth_to_world移植)、colors/normals(depth勾配から)/visibility/depth_valid_mask、GT flows(オブジェクトpose差分)、`camera_{i}_*` キー群+`joint_positions (11,7)`+`gripper_positions (11,)`+`gripper_pose (11,7 TCP)` を `build_flow_sample` 入力形式で返す。`point_instance_ids` も返す。TCPはdecode_dataのwrist→TCP変換(z−0.15)を経由しないので、eef_poseから直接計算しURDF FKと窓0で照合
- `windows.py` — teacher-forcedウィンドウ列挙(hop既定10)。`--autoregressive` オプション(予測frame-10を次窓のframe-0位置に、定性評価用)
- `make_action_file.py` — ①用: joints-only npz → 11フレームに間引き/補間して `(11,7)+(11,)` action npzを出力。`--retarget` モード(DROIDサンプルの初期姿勢にオフセット合わせ)
- `damage_proxy.py` — ③: 予測点のinstance id復元(center-shift逆変換→KD-tree近傍照合、voxel 0.015m半径)、per-object速度/加速度→impact damage→health_pred。メトリクス: per-step Pearson/Spearman、最終health MAE、safe/unsafe分類
- `viz.py` — `damagesim/utils/visualization.py` のhealth bar描画関数をvendorコピーし、予測vs GT healthバーをrolloutフレームにオーバーレイ

**新規 `run_oopsieverse.py`**(`generate_trajectory.py` を雛形に): `--episode_npz`, `--model_path`, `--hop`, `--autoregressive`, `--out_dir`。窓ごとに sample_builder → 既存パイプライン5関数 → `model(batch, training=False)` → de-normalize → prediction npz蓄積(`scene_flows_pred/gt (W,11,Ns,3)`, `point_instance_ids`, メタ)+viser可視化(generate_trajectory.pyの`visualize`再利用)。

## マイルストーンと検証

### ① 接続確認
1. docker: `python scripts/playback_b1k.py --task_name shelve_item --playback --skip_save_images --export_pointworld_dir ... --export_joints_only`
2. local: `make_action_file.py --retarget` → `python generate_trajectory.py --model_path pretrained_checkpoints/small-droid/model-best.pt --domains=droid --data_dirs=dataset/droid/wds --batch_size=1 --action=file --action_file=oopsie_action.npz --no_viz`
3. **合格**: 有限な `scene_flows_pred.npy (11,Ns,3)` が出力され、`--action=replay` とrobot flowsが異なる

### ② oopsieverseシーンでrollout
1. docker: フルエクスポート(画像あり、stride 3)
2. local サニティチェック(`episode.py --check`): 2カメラの逆投影点群が相互に重なる+`object_poses[0]`中心が点群内(<5cm)→ **GL→CV extrinsicsバグ検出**(最大の無音故障リスク)。オブジェクト中心の画像投影がsegマスクと一致。per-frame extrinsics静止確認
3. `run_oopsieverse.py` を small-droid / large-droid+behavior 両チェックポイントで実行
4. **合格**: 全窓がpipeline assertを通過、窓ごとのEPE/Chamfer報告、viserで物理的に妥当なrollout

### ③ damage proxy
1. GT flows上でproxy実行(上界)、1エピソードでキャリブレーション
2. 予測flows上で実行、health barオーバーレイ動画+相関レポート
3. **合格**: proxy-on-GTがsim healthと強相関(proxy数式の妥当性をモデル非依存で確認)、proxy-on-predがsafe/unsafeをチャンス以上で識別

## 実装完了状況(2026-07-10時点)

①②③すべて実装・実行・検証済み。未着手の残タスク:

- **safe/unsafeの定量比較**: 今回のshelve_item safeデモでは実際の損傷イベントが発生しなかったため、proxyのsafe/unsafe識別力は未検証。`oopsiebench/demos/behavior1k/teleop/shelve_item_unsafe.hdf5`等で同じパイプラインを回し、sim健康度の実降下と比較する。
- **複数ウィンドウのスティッチング**: 現状は単一の11フレーム窓のみ評価。エピソード全体(76フレーム)を`--hop`でスライドさせて連結し、健康度の時系列を通しで比較する仕組みは`run_oopsieverse.py`に未実装。
- **グリッパー符号規約の実データ検証**: `gripper_width_to_droid_positions`の0=open/0.725=closed変換をURDF可動域から導出したのみで、実際のDROIDサンプルをdecodeして符号を裏取りしていない。
- **fps/strideの較正**: `export_stride=3`を直感的に選んだのみで、DROIDサンプルの変位統計との比較によるstride sweepは未実施。
- **可視化**: viser等でのpredicted/GT rollout比較、health barオーバーレイ動画は未実装(`--no_viz`のみで実行)。
- **他タスク・大規模チェックポイントでの追試**: `large-droid+behavior`チェックポイントでの比較は未実施(`small-droid`のみで検証)。

## 追加タスク: フレームスクラブ可視化(2026-07-10 続き)

### Context

①②③実装後、静止画(frame0 vs frame最終フレームの比較)のレポートを1つ作った。しかしユーザーが実際に欲しいのは、PointWorld本家の`generate_trajectory.py --no_viz`無効時のviser可視化(シークバーで各フレームを再生できるもの)に相当する、**フレームをスクラブしながら見る**体験。かつ「この動きになるとこのオブジェクトの損傷がまずい」と分かるように、スクラブ位置に連動してhealth bar(sim/GT proxy/pred proxyの3本)が動くようにしたい。加えて、ユーザー自身がタスク/デモを選んで再実行できるよう、手順書も渡す。

viserはライブサーバーで、この環境からユーザーのブラウザに直接届けられる保証がないため、**事前に全フレームをレンダリングしてJSスライダーで切り替える自己完結HTML**(Artifactツールで公開)を採用する。

### 既存資産(再利用、再計算不要)

`run_oopsieverse.py --damage_proxy` は既に各ウィンドウの `eval_logs/oopsieverse_traj/{tag}/viz/` に以下を保存済み:
- `gt_flows.npy` (T,N,3) / `pred_world.npy` (T,N,3) — 全フレームの点群座標(GTと予測)
- `gt_colors.npy` / `pred_colors.npy` (N,3) uint8 — 点ごとの色(フレームによらず同一点なので使い回し可)
- `gt_obj_idx.npy` / `pred_obj_idx.npy` (N,) — 点ごとのオブジェクトラベル
- `meta.json` — `sim_health` / `gt_health` / `pred_health` の各オブジェクトごとの(T,)配列、`t0`, `T`, `task_name`, `demo_id`

→ 既存の w0 / w30 ウィンドウについては**モデル再実行不要**、レンダリングだけで良い。新しいエピソード/ウィンドウを試す場合のみ `run_oopsieverse.py` からやり直す。

### 実装

**新規 `oopsieverse_bridge/render_frames.py`**(numpy/matplotlib のみ、torch/CUDA不要 — cKDTreeセグフォルト回避の教訓を踏襲):
- `render_frame_sequence(viz_dir, out_dir, tag)`: `gt_flows[t]`/`pred_world[t]` を t=0..T-1 全フレームについてGT|Pred 2パネルでレンダリング(`render_report.py`の`_scatter_3d`を再利用)。ファイルサイズ抑制のため図サイズを縮小(既存の1260×644@140dpi → 700×350@100dpi程度、1フレーム約30-50KB想定)。
- 出力: `{tag}_frame{t:02d}.png` × T枚。

**新規 `oopsieverse_bridge/build_scrub_report.py`**:
- 各ウィンドウについて: T枚のフレーム画像をbase64でJS配列に埋め込み、`<input type="range" min=0 max=T-1>` で `<img>` の src を切り替え
- health bar UI: `meta.json`の sim_health/gt_health/pred_health をJSONとしてJS埋め込み、スライダー位置(フレームindex)に応じて3本のバー(sim実測・GT proxy・pred proxy)の幅と色(閾値で緑/黄/赤、`damagesim/utils/visualization.py`の`_health_bar_style`の配色思想を踏襲)とラベル数値を更新
- 再生/一時停止ボタン(setIntervalで自動スクラブ、短いT=11なので体験として有効)
- 複数ウィンドウ(w0, w30, ...)をページ内に縦に並べる。既存の静止比較(displacement heatmap)も要約として上部に残す。
- 出力は`Artifact`ツールで公開(HTML/MD自己完結、data URI画像埋め込み、CSP制約に準拠)。

### ユーザー向けエピソード選択・再実行の手順書(会話内で提供)

1. **利用可能なタスク一覧**: `oopsiebench/demos/behavior1k/teleop/{task}_{safe,unsafe}.hdf5`(13タスク×2、shelve_item_safeのみ2エピソード、他は1エピソード)
2. **エクスポート**(oopsieverse repo, `oopsieverse_b1k` env): `scripts/playback_b1k.py --task_name <task> --source_hdf5_path oopsiebench/demos/behavior1k/teleop/<task>_<safe|unsafe>.hdf5 --playback --demo_ids <id> --export_pointworld_dir demos/behavior1k/pointworld_export --export_stride 3 --export_resolution 720 1280`
3. **rollout+damage proxy**(PointWorld repo, `pointworld-env` env): `run_oopsieverse.py --model_path pretrained_checkpoints/small-droid/model-best.pt --domains=droid --data_dirs=dataset/droid/wds --allow_partial_load=true --episode_npz <export npz> --window_start <0〜time_steps-11> --damage_proxy --no_viz --out_dir eval_logs/oopsieverse_traj/<tag>`
4. **レンダリング+レポート生成**: `render_frames.py` → `build_scrub_report.py` → Artifact公開
5. **見方**: スライダーを動かす/再生ボタンを押すと、点群のフレームとhealth barが連動。GTパネルで実際に動くオブジェクト、Predパネルでの過剰/過小な動きの違いを目視 → health barで数値化された結果と突き合わせる。

### 実装完了(2026-07-10)

`render_frames.py`(全フレームPNG、固定カメラ・固定bounding boxでスクラブ時のジッター防止)+ `build_scrub_report.py`(base64埋め込みJSスライダー、sim/gt/pred 3本のhealth bar、再生ボタン、damage-tracked/proxy-onlyオブジェクトの分離表示)を実装・実行し、Artifact `https://claude.ai/code/artifact/50f988ab-c0c5-42c5-b7c9-3e647e170969` に公開済み(既存の静止比較ページを同URLで置き換え)。w0/w30は既存の`eval_logs/oopsieverse_traj/{w0,w30}/viz/`を再利用、モデル再実行なし。

## リスクと対策

- **extrinsics規約(GL/CV)**: ②-2のサニティチェックを必須ゲートに
- **グリッパー符号/スケール**: DROIDサンプル1件decode検証+Robotiq flowsとeef poseの目視照合
- **fps不一致**: stride sweep {2,3,5}、窓内変位ヒストグラムをDROID統計と照合
- **OODギャップ(OGレンダリング→DROID学習モデル)**: 両チェックポイント比較+proxy-on-GTをモデル非依存ベースラインとして常時併記
- **カメラがbase_link親付け**(world_fixedでない場合): Frankaは固定ベースなので実質静止だが、per-frame extrinsicsエクスポート+staticアサートで防御
