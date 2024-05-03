using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using UnityEngine.InputSystem;

using UnityEngine.Events;

public class BasicCamControl : MonoBehaviour
{
    public Transform camPivot;
    public Camera cam;//unity transform, child of camPivot

    public float lookSpeed=1;
    public float moveSpeed = 1;
    public float mouseWheelMoveSpeed = 1;

    public bool invertVerticalLook = false;
    public float minHeightOverTerrain=1f;
    public float maxHeight;
    public bool goBoundless=true;


    /*bool initComplete = false;
    public void Start()
    {

        initComplete = true;
        //camPivot.position = Vector3.zero;
        
    }*/


    void LateUpdate()
    {
        if (Input.GetMouseButton(0)|| Input.GetMouseButton(1))
        {
            float horiz = Input.GetAxis("Mouse X") * lookSpeed;
            float vert = Input.GetAxis("Mouse Y") * lookSpeed;
            if (!invertVerticalLook) vert = -vert;
            Vector3 lookEulsers = camPivot.rotation.eulerAngles;
            lookEulsers.x += vert * Time.deltaTime;
            lookEulsers.y += horiz * Time.deltaTime;
            camPivot.rotation = Quaternion.Euler(lookEulsers);
        }
        float horizStrafe = Input.GetAxis("Horizontal")* moveSpeed;
        float forwardStrafe = Input.GetAxis("Vertical")* moveSpeed;

        Vector3 camHorizFormward = camPivot.forward;
      //  camHorizFormward.y = 0;
        camHorizFormward.Normalize();

        camPivot.position += Time.deltaTime * camHorizFormward * forwardStrafe;
        camPivot.position += Time.deltaTime * camPivot.right * horizStrafe;

        float zoom = Input.GetAxis("Mouse ScrollWheel") * mouseWheelMoveSpeed;
        camPivot.position += camPivot.forward * zoom * Time.deltaTime;

    }
    public float focusAtDefaultDistance = 2;
    Coroutine currentMoveProcess = null;
    public void FocusCameraOnPosition(Vector3 pos)
    {
        Vector3 directionOfDestinationFromStart = (pos - camPivot.position).normalized;
        Vector3 camDestinationPos = pos - directionOfDestinationFromStart*focusAtDefaultDistance*2;
        Quaternion destinationOrientation = Quaternion.LookRotation((pos-camDestinationPos).normalized, Vector3.up);
#if UNITY_EDITOR
        Debug.DrawLine(camDestinationPos, pos, Color.blue,20);
        Debug.DrawLine(camDestinationPos, camDestinationPos + (destinationOrientation*Vector3.forward*20f), Color.green,20);
#endif
        if (currentMoveProcess == null)// run only if not already running one.
            currentMoveProcess =StartCoroutine(MoveFocusTo(camDestinationPos, destinationOrientation));

    }
    public float reFocusSpeed = 1;
    public float reFocusRotSpeed = 1;
    IEnumerator MoveFocusTo(Vector3 destinationPos, Quaternion destinationOrientation)
    {
        while ((transform.position - destinationPos).sqrMagnitude > 0.1f || ( Quaternion.Angle(transform.rotation, destinationOrientation)>1) )
        {
            Debug.DrawLine(destinationPos, destinationPos + (destinationOrientation * Vector3.forward*20f), Color.green, 20);
            Debug.Log("focus cam destination - not there yet");
            //  Debug.Log("current pos-> dest pos: " + transform.position + "->" + destinationPos);
            // Debug.Log("current rot-> dest rot: " + transform.rotation + "->" + destinationOrientation);

            transform.position = Vector3.MoveTowards(transform.position, destinationPos, reFocusSpeed * Time.fixedDeltaTime);//.unscaledDeltaTime);
            transform.rotation = Quaternion.RotateTowards(transform.rotation, destinationOrientation, reFocusRotSpeed * Time.fixedDeltaTime);//.unscaledDeltaTime);
            yield return 0;
        }
        Debug.Log("focus cam destination reached");
        currentMoveProcess = null;
    }
}
