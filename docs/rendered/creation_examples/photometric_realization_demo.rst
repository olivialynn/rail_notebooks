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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f0364636740>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>28.423994</td>
          <td>1.341139</td>
          <td>26.696139</td>
          <td>0.163852</td>
          <td>26.017506</td>
          <td>0.080074</td>
          <td>25.393661</td>
          <td>0.075254</td>
          <td>25.135610</td>
          <td>0.114202</td>
          <td>25.212489</td>
          <td>0.265576</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.444550</td>
          <td>0.646399</td>
          <td>27.398740</td>
          <td>0.261383</td>
          <td>27.068267</td>
          <td>0.313079</td>
          <td>27.193915</td>
          <td>0.599310</td>
          <td>26.676740</td>
          <td>0.788237</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.980643</td>
          <td>0.541076</td>
          <td>25.977671</td>
          <td>0.087862</td>
          <td>24.758791</td>
          <td>0.026315</td>
          <td>23.894977</td>
          <td>0.020145</td>
          <td>23.150519</td>
          <td>0.019960</td>
          <td>22.854861</td>
          <td>0.034303</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>31.414082</td>
          <td>3.985915</td>
          <td>27.600425</td>
          <td>0.345068</td>
          <td>27.741419</td>
          <td>0.344265</td>
          <td>26.710032</td>
          <td>0.233882</td>
          <td>26.079884</td>
          <td>0.254544</td>
          <td>25.097612</td>
          <td>0.241686</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.616672</td>
          <td>0.183784</td>
          <td>25.900796</td>
          <td>0.082116</td>
          <td>25.405081</td>
          <td>0.046531</td>
          <td>24.785061</td>
          <td>0.043864</td>
          <td>24.384584</td>
          <td>0.058931</td>
          <td>23.591214</td>
          <td>0.065891</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.214410</td>
          <td>0.300872</td>
          <td>26.264657</td>
          <td>0.112945</td>
          <td>25.954348</td>
          <td>0.075730</td>
          <td>26.238317</td>
          <td>0.157182</td>
          <td>25.726554</td>
          <td>0.189683</td>
          <td>25.882447</td>
          <td>0.449887</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.696974</td>
          <td>0.438509</td>
          <td>26.949106</td>
          <td>0.202939</td>
          <td>26.507251</td>
          <td>0.122976</td>
          <td>26.019761</td>
          <td>0.130226</td>
          <td>27.274010</td>
          <td>0.633999</td>
          <td>26.040188</td>
          <td>0.505980</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.904856</td>
          <td>0.511991</td>
          <td>27.317658</td>
          <td>0.275153</td>
          <td>26.713442</td>
          <td>0.146954</td>
          <td>26.538145</td>
          <td>0.202666</td>
          <td>26.088542</td>
          <td>0.256357</td>
          <td>25.448430</td>
          <td>0.321253</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.450416</td>
          <td>0.750058</td>
          <td>27.386228</td>
          <td>0.290863</td>
          <td>26.275887</td>
          <td>0.100503</td>
          <td>25.985840</td>
          <td>0.126455</td>
          <td>25.540268</td>
          <td>0.161938</td>
          <td>25.275887</td>
          <td>0.279637</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.987549</td>
          <td>0.543789</td>
          <td>26.295091</td>
          <td>0.115976</td>
          <td>26.215392</td>
          <td>0.095311</td>
          <td>25.655792</td>
          <td>0.094807</td>
          <td>25.082303</td>
          <td>0.109014</td>
          <td>24.571298</td>
          <td>0.155198</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>28.198962</td>
          <td>1.276971</td>
          <td>26.791845</td>
          <td>0.203868</td>
          <td>25.962444</td>
          <td>0.089715</td>
          <td>25.299081</td>
          <td>0.082010</td>
          <td>25.040241</td>
          <td>0.123329</td>
          <td>25.210185</td>
          <td>0.309085</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.923593</td>
          <td>0.981370</td>
          <td>28.265483</td>
          <td>0.586288</td>
          <td>26.839029</td>
          <td>0.303831</td>
          <td>26.112659</td>
          <td>0.303470</td>
          <td>26.741463</td>
          <td>0.927143</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.324203</td>
          <td>0.763414</td>
          <td>25.872129</td>
          <td>0.094296</td>
          <td>24.750417</td>
          <td>0.031410</td>
          <td>23.884903</td>
          <td>0.024111</td>
          <td>23.137383</td>
          <td>0.023639</td>
          <td>22.817764</td>
          <td>0.040222</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.747589</td>
          <td>0.461919</td>
          <td>27.095694</td>
          <td>0.252362</td>
          <td>26.763684</td>
          <td>0.304640</td>
          <td>25.910893</td>
          <td>0.274169</td>
          <td>25.569040</td>
          <td>0.434772</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.411340</td>
          <td>0.390457</td>
          <td>25.832907</td>
          <td>0.089313</td>
          <td>25.521295</td>
          <td>0.060777</td>
          <td>24.861925</td>
          <td>0.055718</td>
          <td>24.317386</td>
          <td>0.065389</td>
          <td>23.648862</td>
          <td>0.082068</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.175992</td>
          <td>0.329373</td>
          <td>26.246701</td>
          <td>0.130504</td>
          <td>26.286383</td>
          <td>0.121597</td>
          <td>26.087656</td>
          <td>0.166261</td>
          <td>26.272206</td>
          <td>0.351025</td>
          <td>25.052687</td>
          <td>0.277661</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.177808</td>
          <td>0.684603</td>
          <td>26.949858</td>
          <td>0.233365</td>
          <td>27.244699</td>
          <td>0.268847</td>
          <td>26.283901</td>
          <td>0.193072</td>
          <td>25.647006</td>
          <td>0.207853</td>
          <td>26.016154</td>
          <td>0.572711</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.548897</td>
          <td>0.877972</td>
          <td>27.248559</td>
          <td>0.299921</td>
          <td>26.802503</td>
          <td>0.187748</td>
          <td>26.262981</td>
          <td>0.191327</td>
          <td>25.845388</td>
          <td>0.247056</td>
          <td>25.219175</td>
          <td>0.315058</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.589141</td>
          <td>2.417673</td>
          <td>27.386292</td>
          <td>0.339872</td>
          <td>26.524230</td>
          <td>0.150848</td>
          <td>25.679632</td>
          <td>0.118234</td>
          <td>25.566742</td>
          <td>0.199440</td>
          <td>25.757600</td>
          <td>0.485449</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.167335</td>
          <td>0.324648</td>
          <td>26.440531</td>
          <td>0.152698</td>
          <td>25.969197</td>
          <td>0.091161</td>
          <td>25.786524</td>
          <td>0.126925</td>
          <td>25.490990</td>
          <td>0.183307</td>
          <td>24.640027</td>
          <td>0.195271</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.616911</td>
          <td>0.153142</td>
          <td>26.134902</td>
          <td>0.088813</td>
          <td>25.441456</td>
          <td>0.078511</td>
          <td>25.016241</td>
          <td>0.102911</td>
          <td>24.624683</td>
          <td>0.162468</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.861095</td>
          <td>0.422688</td>
          <td>28.059269</td>
          <td>0.440536</td>
          <td>27.547308</td>
          <td>0.454612</td>
          <td>26.328058</td>
          <td>0.311518</td>
          <td>25.234155</td>
          <td>0.270556</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.500244</td>
          <td>0.809639</td>
          <td>26.010207</td>
          <td>0.097155</td>
          <td>24.843344</td>
          <td>0.030757</td>
          <td>23.884528</td>
          <td>0.021709</td>
          <td>23.148480</td>
          <td>0.021583</td>
          <td>22.783080</td>
          <td>0.035073</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.845956</td>
          <td>0.562797</td>
          <td>28.917818</td>
          <td>1.017193</td>
          <td>27.489019</td>
          <td>0.345259</td>
          <td>27.066939</td>
          <td>0.385699</td>
          <td>26.250102</td>
          <td>0.358344</td>
          <td>25.224719</td>
          <td>0.331732</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.408382</td>
          <td>0.351294</td>
          <td>25.756794</td>
          <td>0.072413</td>
          <td>25.450050</td>
          <td>0.048496</td>
          <td>24.803472</td>
          <td>0.044654</td>
          <td>24.490182</td>
          <td>0.064809</td>
          <td>23.604049</td>
          <td>0.066744</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.438292</td>
          <td>0.377576</td>
          <td>26.468325</td>
          <td>0.144195</td>
          <td>26.190509</td>
          <td>0.100881</td>
          <td>26.274665</td>
          <td>0.175611</td>
          <td>25.261867</td>
          <td>0.137699</td>
          <td>25.482894</td>
          <td>0.355266</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.146359</td>
          <td>0.614712</td>
          <td>26.923662</td>
          <td>0.201389</td>
          <td>26.946211</td>
          <td>0.182129</td>
          <td>26.533354</td>
          <td>0.205210</td>
          <td>25.901115</td>
          <td>0.223024</td>
          <td>26.416541</td>
          <td>0.670670</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.605693</td>
          <td>0.421504</td>
          <td>27.374917</td>
          <td>0.299732</td>
          <td>26.965942</td>
          <td>0.191057</td>
          <td>26.147129</td>
          <td>0.152818</td>
          <td>26.457591</td>
          <td>0.360384</td>
          <td>25.546078</td>
          <td>0.363180</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.914999</td>
          <td>1.785394</td>
          <td>27.113056</td>
          <td>0.255711</td>
          <td>26.750305</td>
          <td>0.169572</td>
          <td>25.719224</td>
          <td>0.112920</td>
          <td>25.441858</td>
          <td>0.166414</td>
          <td>25.710927</td>
          <td>0.437630</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.189237</td>
          <td>0.641407</td>
          <td>26.479821</td>
          <td>0.140710</td>
          <td>26.024216</td>
          <td>0.083762</td>
          <td>25.649106</td>
          <td>0.098174</td>
          <td>25.243980</td>
          <td>0.130386</td>
          <td>24.786472</td>
          <td>0.193744</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
