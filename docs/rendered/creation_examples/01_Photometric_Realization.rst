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

    <pzflow.flow.Flow at 0x7f0940f06a10>



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
    0      23.994413  0.012811  0.010473  
    1      25.391064  0.084057  0.063360  
    2      24.304707  0.029027  0.024822  
    3      25.291103  0.005913  0.005303  
    4      25.096743  0.233677  0.168255  
    ...          ...       ...       ...  
    99995  24.737946  0.028804  0.020974  
    99996  24.224169  0.105613  0.092030  
    99997  25.613836  0.012327  0.008804  
    99998  25.274899  0.195688  0.131078  
    99999  25.699642  0.017340  0.013564  
    
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
          <td>28.433849</td>
          <td>1.348136</td>
          <td>26.802671</td>
          <td>0.179378</td>
          <td>25.928146</td>
          <td>0.073997</td>
          <td>25.232239</td>
          <td>0.065234</td>
          <td>24.618894</td>
          <td>0.072528</td>
          <td>24.016931</td>
          <td>0.095927</td>
          <td>0.012811</td>
          <td>0.010473</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.988146</td>
          <td>1.769075</td>
          <td>27.241344</td>
          <td>0.258551</td>
          <td>26.845400</td>
          <td>0.164533</td>
          <td>26.531278</td>
          <td>0.201501</td>
          <td>25.793796</td>
          <td>0.200729</td>
          <td>25.980336</td>
          <td>0.484078</td>
          <td>0.084057</td>
          <td>0.063360</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.109501</td>
          <td>0.508751</td>
          <td>30.602366</td>
          <td>1.980933</td>
          <td>25.865032</td>
          <td>0.113850</td>
          <td>24.957878</td>
          <td>0.097769</td>
          <td>24.288488</td>
          <td>0.121598</td>
          <td>0.029027</td>
          <td>0.024822</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.136542</td>
          <td>1.011547</td>
          <td>27.110247</td>
          <td>0.205837</td>
          <td>26.304174</td>
          <td>0.166276</td>
          <td>25.647494</td>
          <td>0.177412</td>
          <td>25.685314</td>
          <td>0.386973</td>
          <td>0.005913</td>
          <td>0.005303</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.289730</td>
          <td>0.319545</td>
          <td>26.071619</td>
          <td>0.095414</td>
          <td>26.002263</td>
          <td>0.079004</td>
          <td>25.673610</td>
          <td>0.096301</td>
          <td>25.168492</td>
          <td>0.117518</td>
          <td>24.917059</td>
          <td>0.208012</td>
          <td>0.233677</td>
          <td>0.168255</td>
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
          <td>26.522561</td>
          <td>0.383684</td>
          <td>26.513460</td>
          <td>0.140107</td>
          <td>25.422186</td>
          <td>0.047243</td>
          <td>25.053658</td>
          <td>0.055677</td>
          <td>24.825511</td>
          <td>0.087035</td>
          <td>24.885989</td>
          <td>0.202666</td>
          <td>0.028804</td>
          <td>0.020974</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.314140</td>
          <td>0.684261</td>
          <td>26.517673</td>
          <td>0.140616</td>
          <td>25.891738</td>
          <td>0.071652</td>
          <td>25.197966</td>
          <td>0.063282</td>
          <td>24.955600</td>
          <td>0.097574</td>
          <td>24.290634</td>
          <td>0.121824</td>
          <td>0.105613</td>
          <td>0.092030</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.045454</td>
          <td>0.566949</td>
          <td>26.858145</td>
          <td>0.187991</td>
          <td>26.583582</td>
          <td>0.131386</td>
          <td>26.309886</td>
          <td>0.167087</td>
          <td>25.681779</td>
          <td>0.182641</td>
          <td>25.568654</td>
          <td>0.353313</td>
          <td>0.012327</td>
          <td>0.008804</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.680461</td>
          <td>0.433058</td>
          <td>26.111888</td>
          <td>0.098840</td>
          <td>26.187330</td>
          <td>0.092991</td>
          <td>25.853683</td>
          <td>0.112730</td>
          <td>26.237873</td>
          <td>0.289480</td>
          <td>24.987479</td>
          <td>0.220606</td>
          <td>0.195688</td>
          <td>0.131078</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.109210</td>
          <td>0.593311</td>
          <td>26.968535</td>
          <td>0.206270</td>
          <td>26.590616</td>
          <td>0.132188</td>
          <td>26.373742</td>
          <td>0.176411</td>
          <td>25.999983</td>
          <td>0.238342</td>
          <td>26.212728</td>
          <td>0.573473</td>
          <td>0.017340</td>
          <td>0.013564</td>
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
          <td>27.631028</td>
          <td>0.918138</td>
          <td>26.496651</td>
          <td>0.158859</td>
          <td>25.978027</td>
          <td>0.090991</td>
          <td>25.126264</td>
          <td>0.070433</td>
          <td>24.641330</td>
          <td>0.087054</td>
          <td>23.992090</td>
          <td>0.110906</td>
          <td>0.012811</td>
          <td>0.010473</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.075913</td>
          <td>0.566433</td>
          <td>26.612087</td>
          <td>0.160591</td>
          <td>25.883068</td>
          <td>0.139159</td>
          <td>25.485469</td>
          <td>0.183938</td>
          <td>25.336168</td>
          <td>0.347519</td>
          <td>0.084057</td>
          <td>0.063360</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.647473</td>
          <td>1.607322</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.447955</td>
          <td>0.667326</td>
          <td>25.914879</td>
          <td>0.140720</td>
          <td>24.954808</td>
          <td>0.114780</td>
          <td>24.350270</td>
          <td>0.151504</td>
          <td>0.029027</td>
          <td>0.024822</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.477549</td>
          <td>0.833102</td>
          <td>29.234707</td>
          <td>1.176866</td>
          <td>27.298599</td>
          <td>0.279828</td>
          <td>26.098120</td>
          <td>0.164264</td>
          <td>25.446316</td>
          <td>0.174811</td>
          <td>26.910028</td>
          <td>1.027004</td>
          <td>0.005913</td>
          <td>0.005303</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.559008</td>
          <td>0.214718</td>
          <td>26.187898</td>
          <td>0.135912</td>
          <td>26.020484</td>
          <td>0.106683</td>
          <td>25.622150</td>
          <td>0.123415</td>
          <td>25.312622</td>
          <td>0.175643</td>
          <td>25.009791</td>
          <td>0.295107</td>
          <td>0.233677</td>
          <td>0.168255</td>
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
          <td>26.323230</td>
          <td>0.365079</td>
          <td>26.448746</td>
          <td>0.152709</td>
          <td>25.378488</td>
          <td>0.053644</td>
          <td>25.140776</td>
          <td>0.071469</td>
          <td>24.677653</td>
          <td>0.090032</td>
          <td>24.628966</td>
          <td>0.191960</td>
          <td>0.028804</td>
          <td>0.020974</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.933999</td>
          <td>0.587600</td>
          <td>26.814761</td>
          <td>0.213666</td>
          <td>25.871782</td>
          <td>0.085562</td>
          <td>25.145039</td>
          <td>0.074030</td>
          <td>24.774862</td>
          <td>0.101045</td>
          <td>24.208120</td>
          <td>0.138154</td>
          <td>0.105613</td>
          <td>0.092030</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.128140</td>
          <td>0.660174</td>
          <td>26.750629</td>
          <td>0.196998</td>
          <td>26.427278</td>
          <td>0.134609</td>
          <td>26.468369</td>
          <td>0.224452</td>
          <td>26.685417</td>
          <td>0.473307</td>
          <td>25.420545</td>
          <td>0.365191</td>
          <td>0.012327</td>
          <td>0.008804</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.358691</td>
          <td>0.397291</td>
          <td>26.112231</td>
          <td>0.123085</td>
          <td>25.978643</td>
          <td>0.099104</td>
          <td>25.663366</td>
          <td>0.123161</td>
          <td>25.495111</td>
          <td>0.197714</td>
          <td>25.729046</td>
          <td>0.497902</td>
          <td>0.195688</td>
          <td>0.131078</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.256757</td>
          <td>0.346232</td>
          <td>26.657164</td>
          <td>0.182139</td>
          <td>26.799945</td>
          <td>0.185195</td>
          <td>26.190550</td>
          <td>0.177825</td>
          <td>25.723697</td>
          <td>0.220882</td>
          <td>25.296839</td>
          <td>0.331429</td>
          <td>0.017340</td>
          <td>0.013564</td>
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
          <td>27.071181</td>
          <td>0.578080</td>
          <td>26.957737</td>
          <td>0.204727</td>
          <td>26.141365</td>
          <td>0.089474</td>
          <td>25.154955</td>
          <td>0.061034</td>
          <td>24.723564</td>
          <td>0.079703</td>
          <td>23.825676</td>
          <td>0.081227</td>
          <td>0.012811</td>
          <td>0.010473</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.006716</td>
          <td>0.574024</td>
          <td>27.662532</td>
          <td>0.382414</td>
          <td>26.635847</td>
          <td>0.147161</td>
          <td>26.026299</td>
          <td>0.140687</td>
          <td>25.940369</td>
          <td>0.242206</td>
          <td>25.054056</td>
          <td>0.249475</td>
          <td>0.084057</td>
          <td>0.063360</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.501855</td>
          <td>1.253808</td>
          <td>28.516830</td>
          <td>0.619756</td>
          <td>26.184142</td>
          <td>0.151568</td>
          <td>25.057110</td>
          <td>0.107687</td>
          <td>24.201609</td>
          <td>0.113891</td>
          <td>0.029027</td>
          <td>0.024822</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.279255</td>
          <td>0.668258</td>
          <td>27.661852</td>
          <td>0.362243</td>
          <td>27.436858</td>
          <td>0.269751</td>
          <td>26.650777</td>
          <td>0.222756</td>
          <td>25.578230</td>
          <td>0.167337</td>
          <td>24.941153</td>
          <td>0.212335</td>
          <td>0.005913</td>
          <td>0.005303</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.664527</td>
          <td>0.249523</td>
          <td>26.288406</td>
          <td>0.159505</td>
          <td>26.024015</td>
          <td>0.115979</td>
          <td>25.603548</td>
          <td>0.131792</td>
          <td>25.283515</td>
          <td>0.185278</td>
          <td>25.152024</td>
          <td>0.356061</td>
          <td>0.233677</td>
          <td>0.168255</td>
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
          <td>29.013003</td>
          <td>1.794547</td>
          <td>26.330261</td>
          <td>0.120456</td>
          <td>25.474995</td>
          <td>0.049941</td>
          <td>24.992086</td>
          <td>0.053196</td>
          <td>24.956989</td>
          <td>0.098529</td>
          <td>24.569496</td>
          <td>0.156306</td>
          <td>0.028804</td>
          <td>0.020974</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.826412</td>
          <td>0.202086</td>
          <td>26.110868</td>
          <td>0.097788</td>
          <td>25.161949</td>
          <td>0.069380</td>
          <td>24.859883</td>
          <td>0.100871</td>
          <td>24.282898</td>
          <td>0.136414</td>
          <td>0.105613</td>
          <td>0.092030</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.719147</td>
          <td>0.446329</td>
          <td>26.689253</td>
          <td>0.163106</td>
          <td>26.233218</td>
          <td>0.096964</td>
          <td>26.285547</td>
          <td>0.163915</td>
          <td>26.007538</td>
          <td>0.240184</td>
          <td>25.844445</td>
          <td>0.437760</td>
          <td>0.012327</td>
          <td>0.008804</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.154911</td>
          <td>0.345132</td>
          <td>26.221642</td>
          <td>0.138349</td>
          <td>26.205244</td>
          <td>0.123699</td>
          <td>26.010056</td>
          <td>0.169999</td>
          <td>25.724189</td>
          <td>0.244800</td>
          <td>26.862815</td>
          <td>1.076750</td>
          <td>0.195688</td>
          <td>0.131078</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.275405</td>
          <td>0.667522</td>
          <td>27.121955</td>
          <td>0.234990</td>
          <td>26.680625</td>
          <td>0.143324</td>
          <td>26.264977</td>
          <td>0.161346</td>
          <td>26.510046</td>
          <td>0.360557</td>
          <td>26.321789</td>
          <td>0.621230</td>
          <td>0.017340</td>
          <td>0.013564</td>
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
