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

    <pzflow.flow.Flow at 0x7f8a50384730>



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
    0      23.994413  0.032987  0.021247  
    1      25.391064  0.077408  0.046331  
    2      24.304707  0.106086  0.101945  
    3      25.291103  0.026339  0.022416  
    4      25.096743  0.123587  0.106660  
    ...          ...       ...       ...  
    99995  24.737946  0.116640  0.071013  
    99996  24.224169  0.024052  0.023083  
    99997  25.613836  0.003837  0.002541  
    99998  25.274899  0.030168  0.021039  
    99999  25.699642  0.061725  0.031155  
    
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
          <td>27.983017</td>
          <td>1.047915</td>
          <td>26.732669</td>
          <td>0.169030</td>
          <td>26.120514</td>
          <td>0.087684</td>
          <td>25.224395</td>
          <td>0.064783</td>
          <td>24.749279</td>
          <td>0.081380</td>
          <td>24.113507</td>
          <td>0.104396</td>
          <td>0.032987</td>
          <td>0.021247</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>32.769901</td>
          <td>5.321837</td>
          <td>27.502403</td>
          <td>0.319270</td>
          <td>26.637556</td>
          <td>0.137658</td>
          <td>26.421789</td>
          <td>0.183741</td>
          <td>26.226278</td>
          <td>0.286780</td>
          <td>25.251175</td>
          <td>0.274080</td>
          <td>0.077408</td>
          <td>0.046331</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.120509</td>
          <td>1.876713</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.117393</td>
          <td>0.459889</td>
          <td>25.789793</td>
          <td>0.106615</td>
          <td>25.051773</td>
          <td>0.106145</td>
          <td>24.299088</td>
          <td>0.122722</td>
          <td>0.106086</td>
          <td>0.101945</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>33.787704</td>
          <td>5.128547</td>
          <td>27.260298</td>
          <td>0.233239</td>
          <td>26.382103</td>
          <td>0.177667</td>
          <td>25.472845</td>
          <td>0.152861</td>
          <td>25.516277</td>
          <td>0.339025</td>
          <td>0.026339</td>
          <td>0.022416</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.228126</td>
          <td>0.304200</td>
          <td>26.133130</td>
          <td>0.100695</td>
          <td>25.905488</td>
          <td>0.072529</td>
          <td>25.724065</td>
          <td>0.100656</td>
          <td>25.658893</td>
          <td>0.179135</td>
          <td>25.441952</td>
          <td>0.319599</td>
          <td>0.123587</td>
          <td>0.106660</td>
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
          <td>27.320620</td>
          <td>0.687293</td>
          <td>26.353603</td>
          <td>0.122023</td>
          <td>25.398869</td>
          <td>0.046275</td>
          <td>25.183331</td>
          <td>0.062466</td>
          <td>24.941496</td>
          <td>0.096374</td>
          <td>24.602470</td>
          <td>0.159393</td>
          <td>0.116640</td>
          <td>0.071013</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.732735</td>
          <td>0.169039</td>
          <td>26.058131</td>
          <td>0.082996</td>
          <td>25.120150</td>
          <td>0.059061</td>
          <td>24.866270</td>
          <td>0.090212</td>
          <td>24.212421</td>
          <td>0.113811</td>
          <td>0.024052</td>
          <td>0.023083</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.113093</td>
          <td>0.594946</td>
          <td>26.556685</td>
          <td>0.145415</td>
          <td>26.326981</td>
          <td>0.105099</td>
          <td>26.420220</td>
          <td>0.183497</td>
          <td>25.680340</td>
          <td>0.182418</td>
          <td>26.613677</td>
          <td>0.756151</td>
          <td>0.003837</td>
          <td>0.002541</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.297804</td>
          <td>0.321605</td>
          <td>26.332933</td>
          <td>0.119853</td>
          <td>26.029362</td>
          <td>0.080916</td>
          <td>25.791559</td>
          <td>0.106780</td>
          <td>25.584528</td>
          <td>0.168167</td>
          <td>25.636894</td>
          <td>0.372689</td>
          <td>0.030168</td>
          <td>0.021039</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.748043</td>
          <td>0.455722</td>
          <td>27.147497</td>
          <td>0.239359</td>
          <td>26.667642</td>
          <td>0.141276</td>
          <td>26.008630</td>
          <td>0.128977</td>
          <td>26.178546</td>
          <td>0.275895</td>
          <td>27.419497</td>
          <td>1.233348</td>
          <td>0.061725</td>
          <td>0.031155</td>
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
          <td>26.965229</td>
          <td>0.236032</td>
          <td>26.115541</td>
          <td>0.102873</td>
          <td>24.973838</td>
          <td>0.061678</td>
          <td>24.650398</td>
          <td>0.087940</td>
          <td>23.928409</td>
          <td>0.105139</td>
          <td>0.032987</td>
          <td>0.021247</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.710282</td>
          <td>0.970777</td>
          <td>26.810359</td>
          <td>0.209487</td>
          <td>26.998511</td>
          <td>0.221448</td>
          <td>25.976743</td>
          <td>0.150094</td>
          <td>25.662444</td>
          <td>0.212449</td>
          <td>25.498829</td>
          <td>0.392787</td>
          <td>0.077408</td>
          <td>0.046331</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.002893</td>
          <td>0.498977</td>
          <td>26.151273</td>
          <td>0.178064</td>
          <td>24.837740</td>
          <td>0.107099</td>
          <td>24.178849</td>
          <td>0.135148</td>
          <td>0.106086</td>
          <td>0.101945</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.478372</td>
          <td>1.345446</td>
          <td>28.153742</td>
          <td>0.541873</td>
          <td>26.480272</td>
          <td>0.227046</td>
          <td>25.590577</td>
          <td>0.197835</td>
          <td>24.989497</td>
          <td>0.258991</td>
          <td>0.026339</td>
          <td>0.022416</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.084124</td>
          <td>0.310968</td>
          <td>26.243284</td>
          <td>0.132741</td>
          <td>26.049349</td>
          <td>0.101109</td>
          <td>25.826579</td>
          <td>0.135927</td>
          <td>25.385010</td>
          <td>0.173022</td>
          <td>24.891312</td>
          <td>0.248600</td>
          <td>0.123587</td>
          <td>0.106660</td>
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
          <td>28.365577</td>
          <td>1.414122</td>
          <td>26.494953</td>
          <td>0.162886</td>
          <td>25.408815</td>
          <td>0.056717</td>
          <td>25.054498</td>
          <td>0.068210</td>
          <td>24.800125</td>
          <td>0.103125</td>
          <td>24.485794</td>
          <td>0.174911</td>
          <td>0.116640</td>
          <td>0.071013</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.659380</td>
          <td>1.616135</td>
          <td>26.726909</td>
          <td>0.193357</td>
          <td>26.037355</td>
          <td>0.095993</td>
          <td>25.243708</td>
          <td>0.078252</td>
          <td>25.008685</td>
          <td>0.120216</td>
          <td>24.239305</td>
          <td>0.137632</td>
          <td>0.024052</td>
          <td>0.023083</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.095466</td>
          <td>0.304439</td>
          <td>26.822613</td>
          <td>0.209186</td>
          <td>26.142580</td>
          <td>0.105064</td>
          <td>26.119228</td>
          <td>0.167234</td>
          <td>25.486146</td>
          <td>0.180803</td>
          <td>26.071608</td>
          <td>0.593723</td>
          <td>0.003837</td>
          <td>0.002541</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.989186</td>
          <td>0.279902</td>
          <td>26.146872</td>
          <td>0.117705</td>
          <td>26.070699</td>
          <td>0.098881</td>
          <td>26.027619</td>
          <td>0.155001</td>
          <td>25.870724</td>
          <td>0.249789</td>
          <td>24.778490</td>
          <td>0.217618</td>
          <td>0.030168</td>
          <td>0.021039</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.156892</td>
          <td>0.321519</td>
          <td>26.644386</td>
          <td>0.181318</td>
          <td>26.499157</td>
          <td>0.144295</td>
          <td>26.452928</td>
          <td>0.223259</td>
          <td>26.144264</td>
          <td>0.313475</td>
          <td>25.607273</td>
          <td>0.424732</td>
          <td>0.061725</td>
          <td>0.031155</td>
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
          <td>26.680510</td>
          <td>0.163097</td>
          <td>26.027583</td>
          <td>0.081633</td>
          <td>25.265827</td>
          <td>0.067945</td>
          <td>24.571200</td>
          <td>0.070259</td>
          <td>23.998463</td>
          <td>0.095404</td>
          <td>0.032987</td>
          <td>0.021247</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.299634</td>
          <td>0.696976</td>
          <td>27.399440</td>
          <td>0.306397</td>
          <td>26.893670</td>
          <td>0.180230</td>
          <td>26.198562</td>
          <td>0.160160</td>
          <td>25.610350</td>
          <td>0.180707</td>
          <td>25.546237</td>
          <td>0.364179</td>
          <td>0.077408</td>
          <td>0.046331</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.822298</td>
          <td>0.410664</td>
          <td>25.863343</td>
          <td>0.129779</td>
          <td>25.042916</td>
          <td>0.119615</td>
          <td>24.255961</td>
          <td>0.134748</td>
          <td>0.106086</td>
          <td>0.101945</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.274983</td>
          <td>0.237887</td>
          <td>26.147233</td>
          <td>0.146576</td>
          <td>25.629044</td>
          <td>0.176019</td>
          <td>25.030285</td>
          <td>0.230408</td>
          <td>0.026339</td>
          <td>0.022416</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.590089</td>
          <td>0.445029</td>
          <td>26.237892</td>
          <td>0.126180</td>
          <td>25.889572</td>
          <td>0.083439</td>
          <td>25.717432</td>
          <td>0.117305</td>
          <td>25.422296</td>
          <td>0.169889</td>
          <td>25.450736</td>
          <td>0.370935</td>
          <td>0.123587</td>
          <td>0.106660</td>
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
          <td>26.351468</td>
          <td>0.360289</td>
          <td>26.263441</td>
          <td>0.124213</td>
          <td>25.444720</td>
          <td>0.053892</td>
          <td>25.000934</td>
          <td>0.059695</td>
          <td>24.736997</td>
          <td>0.089910</td>
          <td>24.397188</td>
          <td>0.149411</td>
          <td>0.116640</td>
          <td>0.071013</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.266045</td>
          <td>0.664758</td>
          <td>26.557336</td>
          <td>0.146425</td>
          <td>25.919819</td>
          <td>0.074011</td>
          <td>25.221217</td>
          <td>0.065116</td>
          <td>24.712185</td>
          <td>0.079355</td>
          <td>24.207704</td>
          <td>0.114224</td>
          <td>0.024052</td>
          <td>0.023083</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.070871</td>
          <td>0.577396</td>
          <td>26.451155</td>
          <td>0.132793</td>
          <td>26.473580</td>
          <td>0.119448</td>
          <td>26.167137</td>
          <td>0.147897</td>
          <td>25.932558</td>
          <td>0.225425</td>
          <td>25.638807</td>
          <td>0.373295</td>
          <td>0.003837</td>
          <td>0.002541</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.076190</td>
          <td>0.270648</td>
          <td>26.208114</td>
          <td>0.108361</td>
          <td>26.129732</td>
          <td>0.089208</td>
          <td>25.703342</td>
          <td>0.099792</td>
          <td>25.562739</td>
          <td>0.166535</td>
          <td>25.768729</td>
          <td>0.416057</td>
          <td>0.030168</td>
          <td>0.021039</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.401945</td>
          <td>2.134458</td>
          <td>26.508890</td>
          <td>0.143315</td>
          <td>26.421468</td>
          <td>0.117709</td>
          <td>26.409447</td>
          <td>0.187625</td>
          <td>26.036607</td>
          <td>0.252891</td>
          <td>25.802236</td>
          <td>0.435329</td>
          <td>0.061725</td>
          <td>0.031155</td>
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
