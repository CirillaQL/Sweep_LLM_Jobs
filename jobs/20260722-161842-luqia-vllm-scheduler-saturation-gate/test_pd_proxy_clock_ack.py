#!/usr/bin/env python3

import unittest

from aiohttp.test_utils import TestClient, TestServer

from pd_proxy import CLOCK_ACKS, app


class ClockAckEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        CLOCK_ACKS.clear()
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_publish_read_and_reject_invalid_ack(self) -> None:
        missing = await self.client.get("/control/clock-ack/neptune/1")
        self.assertEqual(missing.status, 404)

        invalid = await self.client.post(
            "/control/clock-ack",
            json={
                "node_group": "unknown",
                "seq": 1,
                "target_mhz": 1755,
                "rc": 0,
                "observed_mhz": "1755",
            },
        )
        self.assertEqual(invalid.status, 400)

        published = await self.client.post(
            "/control/clock-ack",
            json={
                "node_group": "neptune",
                "seq": 1,
                "target_mhz": 1755,
                "rc": 0,
                "observed_mhz": "1755",
            },
        )
        self.assertEqual(published.status, 200)

        received = await self.client.get("/control/clock-ack/neptune/1")
        self.assertEqual(received.status, 200)
        self.assertEqual(await received.text(), "1 1755 0 1755\n")


if __name__ == "__main__":
    unittest.main()
