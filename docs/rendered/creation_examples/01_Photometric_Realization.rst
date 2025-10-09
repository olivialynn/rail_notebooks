Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f2f778dcca0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.162487  0.095260  
    1      25.391064  0.088640  0.060990  
    2      24.304707  0.099270  0.095883  
    3      25.291103  0.057046  0.036920  
    4      25.096743  0.089342  0.051819  
    ...          ...       ...       ...  
    99995  24.737946  0.023722  0.020970  
    99996  24.224169  0.000082  0.000059  
    99997  25.613836  0.122567  0.119156  
    99998  25.274899  0.065939  0.054312  
    99999  25.699642  0.120138  0.079523  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  input: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>28.507019</td>
          <td>1.400655</td>
          <td>26.943572</td>
          <td>0.202000</td>
          <td>26.102185</td>
          <td>0.086281</td>
          <td>25.195832</td>
          <td>0.063163</td>
          <td>24.879136</td>
          <td>0.091239</td>
          <td>24.143091</td>
          <td>0.107131</td>
          <td>0.162487</td>
          <td>0.095260</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.916594</td>
          <td>0.440572</td>
          <td>26.735828</td>
          <td>0.149807</td>
          <td>26.811015</td>
          <td>0.254176</td>
          <td>26.232339</td>
          <td>0.288189</td>
          <td>25.153566</td>
          <td>0.253072</td>
          <td>0.088640</td>
          <td>0.060990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.810077</td>
          <td>0.944182</td>
          <td>30.750179</td>
          <td>2.230331</td>
          <td>28.135860</td>
          <td>0.466298</td>
          <td>25.885471</td>
          <td>0.115895</td>
          <td>25.105249</td>
          <td>0.111219</td>
          <td>24.236174</td>
          <td>0.116190</td>
          <td>0.099270</td>
          <td>0.095883</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.935648</td>
          <td>0.894300</td>
          <td>26.798120</td>
          <td>0.158021</td>
          <td>26.245632</td>
          <td>0.158169</td>
          <td>25.519465</td>
          <td>0.159085</td>
          <td>25.360771</td>
          <td>0.299483</td>
          <td>0.057046</td>
          <td>0.036920</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.025978</td>
          <td>0.559076</td>
          <td>26.049900</td>
          <td>0.093614</td>
          <td>25.825154</td>
          <td>0.067550</td>
          <td>25.518976</td>
          <td>0.084056</td>
          <td>25.294391</td>
          <td>0.131083</td>
          <td>25.397595</td>
          <td>0.308467</td>
          <td>0.089342</td>
          <td>0.051819</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.596035</td>
          <td>0.406045</td>
          <td>26.259078</td>
          <td>0.112397</td>
          <td>25.465474</td>
          <td>0.049094</td>
          <td>25.059181</td>
          <td>0.055950</td>
          <td>25.084408</td>
          <td>0.109215</td>
          <td>25.195074</td>
          <td>0.261825</td>
          <td>0.023722</td>
          <td>0.020970</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.698530</td>
          <td>0.439025</td>
          <td>26.744294</td>
          <td>0.170709</td>
          <td>26.251810</td>
          <td>0.098405</td>
          <td>25.237240</td>
          <td>0.065524</td>
          <td>24.747403</td>
          <td>0.081246</td>
          <td>24.299086</td>
          <td>0.122722</td>
          <td>0.000082</td>
          <td>0.000059</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.740541</td>
          <td>0.453160</td>
          <td>26.606310</td>
          <td>0.151741</td>
          <td>26.472173</td>
          <td>0.119285</td>
          <td>26.372138</td>
          <td>0.176171</td>
          <td>25.880655</td>
          <td>0.215864</td>
          <td>27.042746</td>
          <td>0.992395</td>
          <td>0.122567</td>
          <td>0.119156</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.432972</td>
          <td>0.357821</td>
          <td>26.110803</td>
          <td>0.098746</td>
          <td>26.056253</td>
          <td>0.082858</td>
          <td>26.090288</td>
          <td>0.138409</td>
          <td>26.109029</td>
          <td>0.260693</td>
          <td>26.029958</td>
          <td>0.502182</td>
          <td>0.065939</td>
          <td>0.054312</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.003308</td>
          <td>0.550018</td>
          <td>26.699001</td>
          <td>0.164252</td>
          <td>26.509137</td>
          <td>0.123178</td>
          <td>26.246168</td>
          <td>0.158241</td>
          <td>25.249012</td>
          <td>0.126031</td>
          <td>25.753248</td>
          <td>0.407779</td>
          <td>0.120138</td>
          <td>0.079523</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.619378</td>
          <td>0.185160</td>
          <td>25.956017</td>
          <td>0.094371</td>
          <td>25.172617</td>
          <td>0.077763</td>
          <td>24.719309</td>
          <td>0.098569</td>
          <td>24.097550</td>
          <td>0.128630</td>
          <td>0.162487</td>
          <td>0.095260</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.981067</td>
          <td>0.602641</td>
          <td>27.116885</td>
          <td>0.271078</td>
          <td>26.677893</td>
          <td>0.169988</td>
          <td>26.266762</td>
          <td>0.193173</td>
          <td>25.960975</td>
          <td>0.273199</td>
          <td>24.620579</td>
          <td>0.193851</td>
          <td>0.088640</td>
          <td>0.060990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>31.777461</td>
          <td>4.499296</td>
          <td>28.158063</td>
          <td>0.606228</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.888014</td>
          <td>0.141596</td>
          <td>24.905401</td>
          <td>0.113145</td>
          <td>24.248919</td>
          <td>0.142967</td>
          <td>0.099270</td>
          <td>0.095883</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.426403</td>
          <td>0.809507</td>
          <td>28.774380</td>
          <td>0.899460</td>
          <td>27.701361</td>
          <td>0.387841</td>
          <td>26.542177</td>
          <td>0.240328</td>
          <td>25.568808</td>
          <td>0.195319</td>
          <td>25.244841</td>
          <td>0.320074</td>
          <td>0.057046</td>
          <td>0.036920</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.859494</td>
          <td>0.254631</td>
          <td>25.942952</td>
          <td>0.099914</td>
          <td>25.927711</td>
          <td>0.088576</td>
          <td>25.569366</td>
          <td>0.105900</td>
          <td>25.580868</td>
          <td>0.199200</td>
          <td>25.455158</td>
          <td>0.381150</td>
          <td>0.089342</td>
          <td>0.051819</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.225803</td>
          <td>0.706262</td>
          <td>26.221976</td>
          <td>0.125562</td>
          <td>25.320465</td>
          <td>0.050929</td>
          <td>25.010175</td>
          <td>0.063636</td>
          <td>24.839982</td>
          <td>0.103757</td>
          <td>25.338254</td>
          <td>0.342748</td>
          <td>0.023722</td>
          <td>0.020970</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.590605</td>
          <td>0.172025</td>
          <td>25.937593</td>
          <td>0.087772</td>
          <td>25.289569</td>
          <td>0.081322</td>
          <td>24.861439</td>
          <td>0.105542</td>
          <td>24.167940</td>
          <td>0.129154</td>
          <td>0.000082</td>
          <td>0.000059</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.428371</td>
          <td>0.828884</td>
          <td>26.699939</td>
          <td>0.196628</td>
          <td>26.369054</td>
          <td>0.134062</td>
          <td>25.912443</td>
          <td>0.146957</td>
          <td>25.524032</td>
          <td>0.195359</td>
          <td>25.344844</td>
          <td>0.359285</td>
          <td>0.122567</td>
          <td>0.119156</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.451586</td>
          <td>0.405957</td>
          <td>26.262686</td>
          <td>0.131292</td>
          <td>26.199896</td>
          <td>0.111808</td>
          <td>25.782693</td>
          <td>0.126790</td>
          <td>25.573400</td>
          <td>0.196917</td>
          <td>25.033409</td>
          <td>0.271079</td>
          <td>0.065939</td>
          <td>0.054312</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.568561</td>
          <td>0.450151</td>
          <td>26.945347</td>
          <td>0.238463</td>
          <td>26.532998</td>
          <td>0.152360</td>
          <td>26.014326</td>
          <td>0.158213</td>
          <td>26.197543</td>
          <td>0.334801</td>
          <td>25.315135</td>
          <td>0.346703</td>
          <td>0.120138</td>
          <td>0.079523</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.011789</td>
          <td>0.250554</td>
          <td>26.075157</td>
          <td>0.101765</td>
          <td>25.250575</td>
          <td>0.080809</td>
          <td>24.681199</td>
          <td>0.092605</td>
          <td>24.092138</td>
          <td>0.124303</td>
          <td>0.162487</td>
          <td>0.095260</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.314573</td>
          <td>0.290943</td>
          <td>26.621829</td>
          <td>0.145773</td>
          <td>26.078747</td>
          <td>0.147576</td>
          <td>25.838898</td>
          <td>0.223240</td>
          <td>25.272250</td>
          <td>0.298673</td>
          <td>0.088640</td>
          <td>0.060990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.086739</td>
          <td>1.051258</td>
          <td>28.546437</td>
          <td>0.686932</td>
          <td>25.961492</td>
          <td>0.139300</td>
          <td>24.963884</td>
          <td>0.110155</td>
          <td>24.329005</td>
          <td>0.141535</td>
          <td>0.099270</td>
          <td>0.095883</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.847137</td>
          <td>0.980087</td>
          <td>27.491047</td>
          <td>0.324036</td>
          <td>27.367920</td>
          <td>0.262151</td>
          <td>26.485302</td>
          <td>0.199835</td>
          <td>25.649562</td>
          <td>0.182976</td>
          <td>25.424130</td>
          <td>0.324186</td>
          <td>0.057046</td>
          <td>0.036920</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.775694</td>
          <td>0.484254</td>
          <td>26.160059</td>
          <td>0.109247</td>
          <td>26.045524</td>
          <td>0.087743</td>
          <td>25.636299</td>
          <td>0.099906</td>
          <td>25.647866</td>
          <td>0.189186</td>
          <td>25.227989</td>
          <td>0.286626</td>
          <td>0.089342</td>
          <td>0.051819</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.484248</td>
          <td>0.769760</td>
          <td>26.304975</td>
          <td>0.117659</td>
          <td>25.308886</td>
          <td>0.043017</td>
          <td>25.183467</td>
          <td>0.062924</td>
          <td>24.749267</td>
          <td>0.081934</td>
          <td>24.485019</td>
          <td>0.145115</td>
          <td>0.023722</td>
          <td>0.020970</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.117617</td>
          <td>0.278275</td>
          <td>26.603945</td>
          <td>0.151434</td>
          <td>26.148456</td>
          <td>0.089867</td>
          <td>25.280797</td>
          <td>0.068102</td>
          <td>24.788270</td>
          <td>0.084226</td>
          <td>24.299586</td>
          <td>0.122775</td>
          <td>0.000082</td>
          <td>0.000059</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.904441</td>
          <td>0.565484</td>
          <td>26.694980</td>
          <td>0.188660</td>
          <td>26.598071</td>
          <td>0.156539</td>
          <td>26.112743</td>
          <td>0.167069</td>
          <td>26.124984</td>
          <td>0.307695</td>
          <td>25.497160</td>
          <td>0.389079</td>
          <td>0.122567</td>
          <td>0.119156</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.113599</td>
          <td>0.285965</td>
          <td>26.331170</td>
          <td>0.124634</td>
          <td>26.130257</td>
          <td>0.092714</td>
          <td>25.888543</td>
          <td>0.122027</td>
          <td>25.757683</td>
          <td>0.203689</td>
          <td>25.092675</td>
          <td>0.251979</td>
          <td>0.065939</td>
          <td>0.054312</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.973388</td>
          <td>0.268456</td>
          <td>27.079456</td>
          <td>0.250270</td>
          <td>26.401926</td>
          <td>0.126577</td>
          <td>26.295207</td>
          <td>0.186555</td>
          <td>25.701556</td>
          <td>0.208642</td>
          <td>25.390759</td>
          <td>0.343792</td>
          <td>0.120138</td>
          <td>0.079523</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
