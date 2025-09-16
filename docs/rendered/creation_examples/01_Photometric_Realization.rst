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

    <pzflow.flow.Flow at 0x7f0c2eb808e0>



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
    0      23.994413  0.095377  0.071400  
    1      25.391064  0.031727  0.020633  
    2      24.304707  0.062149  0.039087  
    3      25.291103  0.016314  0.008331  
    4      25.096743  0.104081  0.076929  
    ...          ...       ...       ...  
    99995  24.737946  0.033405  0.033249  
    99996  24.224169  0.036747  0.026907  
    99997  25.613836  0.165561  0.130966  
    99998  25.274899  0.045226  0.042262  
    99999  25.699642  0.090561  0.051410  
    
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
          <td>27.473053</td>
          <td>0.761402</td>
          <td>26.818443</td>
          <td>0.181789</td>
          <td>26.026932</td>
          <td>0.080743</td>
          <td>25.329133</td>
          <td>0.071080</td>
          <td>24.586849</td>
          <td>0.070500</td>
          <td>24.026683</td>
          <td>0.096751</td>
          <td>0.095377</td>
          <td>0.071400</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.806229</td>
          <td>3.398613</td>
          <td>27.188074</td>
          <td>0.247495</td>
          <td>26.770266</td>
          <td>0.154298</td>
          <td>26.735936</td>
          <td>0.238944</td>
          <td>25.785150</td>
          <td>0.199276</td>
          <td>24.786054</td>
          <td>0.186311</td>
          <td>0.031727</td>
          <td>0.020633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.853933</td>
          <td>0.773619</td>
          <td>25.878303</td>
          <td>0.115174</td>
          <td>24.904792</td>
          <td>0.093319</td>
          <td>24.336449</td>
          <td>0.126764</td>
          <td>0.062149</td>
          <td>0.039087</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.204349</td>
          <td>1.946122</td>
          <td>29.037135</td>
          <td>0.952402</td>
          <td>27.049056</td>
          <td>0.195527</td>
          <td>26.444895</td>
          <td>0.187365</td>
          <td>25.541095</td>
          <td>0.162053</td>
          <td>25.418158</td>
          <td>0.313585</td>
          <td>0.016314</td>
          <td>0.008331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.012038</td>
          <td>0.255344</td>
          <td>26.256299</td>
          <td>0.112126</td>
          <td>25.945427</td>
          <td>0.075136</td>
          <td>25.851915</td>
          <td>0.112556</td>
          <td>25.406298</td>
          <td>0.144370</td>
          <td>24.639877</td>
          <td>0.164566</td>
          <td>0.104081</td>
          <td>0.076929</td>
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
          <td>27.312536</td>
          <td>0.683512</td>
          <td>26.570667</td>
          <td>0.147171</td>
          <td>25.445581</td>
          <td>0.048234</td>
          <td>25.209207</td>
          <td>0.063916</td>
          <td>24.704154</td>
          <td>0.078203</td>
          <td>24.573499</td>
          <td>0.155491</td>
          <td>0.033405</td>
          <td>0.033249</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.622694</td>
          <td>0.839354</td>
          <td>27.121171</td>
          <td>0.234209</td>
          <td>26.002855</td>
          <td>0.079046</td>
          <td>25.290502</td>
          <td>0.068690</td>
          <td>24.843417</td>
          <td>0.088417</td>
          <td>24.181374</td>
          <td>0.110772</td>
          <td>0.036747</td>
          <td>0.026907</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.287149</td>
          <td>0.318889</td>
          <td>26.466781</td>
          <td>0.134580</td>
          <td>26.253676</td>
          <td>0.098566</td>
          <td>26.398278</td>
          <td>0.180120</td>
          <td>25.659138</td>
          <td>0.179172</td>
          <td>25.816479</td>
          <td>0.427965</td>
          <td>0.165561</td>
          <td>0.130966</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.819883</td>
          <td>0.217900</td>
          <td>26.279702</td>
          <td>0.114434</td>
          <td>26.149337</td>
          <td>0.089936</td>
          <td>25.854239</td>
          <td>0.112784</td>
          <td>25.566207</td>
          <td>0.165562</td>
          <td>25.072211</td>
          <td>0.236669</td>
          <td>0.045226</td>
          <td>0.042262</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>30.787493</td>
          <td>3.380703</td>
          <td>26.660089</td>
          <td>0.158887</td>
          <td>26.605815</td>
          <td>0.133936</td>
          <td>26.193308</td>
          <td>0.151236</td>
          <td>25.777409</td>
          <td>0.197984</td>
          <td>25.591157</td>
          <td>0.359606</td>
          <td>0.090561</td>
          <td>0.051410</td>
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
          <td>26.848290</td>
          <td>0.549591</td>
          <td>26.891028</td>
          <td>0.225961</td>
          <td>26.228009</td>
          <td>0.115870</td>
          <td>25.373443</td>
          <td>0.089724</td>
          <td>24.773221</td>
          <td>0.100023</td>
          <td>23.899104</td>
          <td>0.104700</td>
          <td>0.095377</td>
          <td>0.071400</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.611748</td>
          <td>0.395941</td>
          <td>26.890259</td>
          <td>0.200146</td>
          <td>26.219353</td>
          <td>0.182510</td>
          <td>25.339601</td>
          <td>0.159983</td>
          <td>25.054210</td>
          <td>0.273138</td>
          <td>0.031727</td>
          <td>0.020633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.096514</td>
          <td>1.212453</td>
          <td>28.442544</td>
          <td>0.725836</td>
          <td>28.709599</td>
          <td>0.799194</td>
          <td>26.252099</td>
          <td>0.188877</td>
          <td>25.004221</td>
          <td>0.120606</td>
          <td>24.183837</td>
          <td>0.132152</td>
          <td>0.062149</td>
          <td>0.039087</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.122729</td>
          <td>1.104391</td>
          <td>27.084462</td>
          <td>0.234901</td>
          <td>26.734841</td>
          <td>0.279418</td>
          <td>25.525086</td>
          <td>0.186952</td>
          <td>26.742526</td>
          <td>0.927978</td>
          <td>0.016314</td>
          <td>0.008331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.693239</td>
          <td>0.492005</td>
          <td>26.009903</td>
          <td>0.106881</td>
          <td>26.026810</td>
          <td>0.097591</td>
          <td>25.589796</td>
          <td>0.108910</td>
          <td>25.880626</td>
          <td>0.257855</td>
          <td>25.302488</td>
          <td>0.341280</td>
          <td>0.104081</td>
          <td>0.076929</td>
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
          <td>26.738627</td>
          <td>0.501011</td>
          <td>26.434685</td>
          <td>0.151095</td>
          <td>25.570179</td>
          <td>0.063688</td>
          <td>25.171064</td>
          <td>0.073533</td>
          <td>24.649281</td>
          <td>0.087956</td>
          <td>24.605661</td>
          <td>0.188524</td>
          <td>0.033405</td>
          <td>0.033249</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.781534</td>
          <td>0.202721</td>
          <td>25.988933</td>
          <td>0.092149</td>
          <td>25.208476</td>
          <td>0.075984</td>
          <td>24.956084</td>
          <td>0.115026</td>
          <td>24.126921</td>
          <td>0.125091</td>
          <td>0.036747</td>
          <td>0.026907</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.465486</td>
          <td>0.426638</td>
          <td>26.835259</td>
          <td>0.224578</td>
          <td>26.367988</td>
          <td>0.136985</td>
          <td>26.310752</td>
          <td>0.210762</td>
          <td>26.030084</td>
          <td>0.302745</td>
          <td>25.647211</td>
          <td>0.462419</td>
          <td>0.165561</td>
          <td>0.130966</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.689936</td>
          <td>0.484134</td>
          <td>26.215411</td>
          <td>0.125383</td>
          <td>25.944551</td>
          <td>0.088887</td>
          <td>25.714672</td>
          <td>0.118815</td>
          <td>25.548880</td>
          <td>0.191828</td>
          <td>25.066422</td>
          <td>0.276927</td>
          <td>0.045226</td>
          <td>0.042262</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.926130</td>
          <td>1.839542</td>
          <td>26.533369</td>
          <td>0.166461</td>
          <td>26.643894</td>
          <td>0.164938</td>
          <td>26.302452</td>
          <td>0.198817</td>
          <td>26.195056</td>
          <td>0.329384</td>
          <td>25.587231</td>
          <td>0.422031</td>
          <td>0.090561</td>
          <td>0.051410</td>
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
          <td>26.797913</td>
          <td>0.498549</td>
          <td>26.815318</td>
          <td>0.195161</td>
          <td>25.979713</td>
          <td>0.084555</td>
          <td>25.270494</td>
          <td>0.073997</td>
          <td>24.765535</td>
          <td>0.090109</td>
          <td>23.937654</td>
          <td>0.097951</td>
          <td>0.095377</td>
          <td>0.071400</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.410228</td>
          <td>0.298821</td>
          <td>26.803819</td>
          <td>0.160287</td>
          <td>25.951902</td>
          <td>0.124018</td>
          <td>25.669622</td>
          <td>0.182453</td>
          <td>25.289588</td>
          <td>0.285376</td>
          <td>0.031727</td>
          <td>0.020633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.936434</td>
          <td>3.550307</td>
          <td>28.876168</td>
          <td>0.880498</td>
          <td>27.865410</td>
          <td>0.391270</td>
          <td>26.298881</td>
          <td>0.171532</td>
          <td>24.932021</td>
          <td>0.098981</td>
          <td>24.491883</td>
          <td>0.150227</td>
          <td>0.062149</td>
          <td>0.039087</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.085895</td>
          <td>1.113867</td>
          <td>27.788836</td>
          <td>0.400328</td>
          <td>27.260649</td>
          <td>0.233807</td>
          <td>26.337410</td>
          <td>0.171444</td>
          <td>25.021352</td>
          <td>0.103593</td>
          <td>25.096479</td>
          <td>0.241996</td>
          <td>0.016314</td>
          <td>0.008331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.781785</td>
          <td>0.226110</td>
          <td>26.094532</td>
          <td>0.106394</td>
          <td>25.947495</td>
          <td>0.083353</td>
          <td>25.744235</td>
          <td>0.113832</td>
          <td>25.489768</td>
          <td>0.171125</td>
          <td>24.670834</td>
          <td>0.186974</td>
          <td>0.104081</td>
          <td>0.076929</td>
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
          <td>30.377912</td>
          <td>3.004025</td>
          <td>26.382575</td>
          <td>0.126732</td>
          <td>25.392528</td>
          <td>0.046714</td>
          <td>25.068692</td>
          <td>0.057323</td>
          <td>25.046353</td>
          <td>0.107217</td>
          <td>25.189002</td>
          <td>0.264296</td>
          <td>0.033405</td>
          <td>0.033249</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.870030</td>
          <td>0.503096</td>
          <td>26.731341</td>
          <td>0.170821</td>
          <td>26.088944</td>
          <td>0.086476</td>
          <td>25.123326</td>
          <td>0.060108</td>
          <td>24.690964</td>
          <td>0.078384</td>
          <td>24.095446</td>
          <td>0.104247</td>
          <td>0.036747</td>
          <td>0.026907</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.786403</td>
          <td>0.541910</td>
          <td>26.473909</td>
          <td>0.165853</td>
          <td>26.408431</td>
          <td>0.141970</td>
          <td>25.898846</td>
          <td>0.148767</td>
          <td>25.962749</td>
          <td>0.286994</td>
          <td>25.739750</td>
          <td>0.495764</td>
          <td>0.165561</td>
          <td>0.130966</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.403379</td>
          <td>0.736721</td>
          <td>26.420497</td>
          <td>0.132143</td>
          <td>26.078421</td>
          <td>0.086671</td>
          <td>25.877531</td>
          <td>0.118172</td>
          <td>25.719699</td>
          <td>0.193229</td>
          <td>25.503218</td>
          <td>0.343579</td>
          <td>0.045226</td>
          <td>0.042262</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.590369</td>
          <td>0.421491</td>
          <td>26.710941</td>
          <td>0.175699</td>
          <td>26.586444</td>
          <td>0.140736</td>
          <td>26.260945</td>
          <td>0.171604</td>
          <td>26.277152</td>
          <td>0.317709</td>
          <td>25.829310</td>
          <td>0.458884</td>
          <td>0.090561</td>
          <td>0.051410</td>
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
